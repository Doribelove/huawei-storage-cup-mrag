"""Ten-document genuine CPU pilot with frozen source-only manifests and timed diagnostics."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch
import psutil
from evidence_lab.cli import load_evidence
from evidence_lab.demo import EvidenceBrowser
from evidence_lab.lexical import BM25
from evidence_lab.pilot import ContextAssembler,answer_support_state,select_pilot
from evidence_lab.safe_data import reject_answer_fields,sha256_file
from evidence_lab.semantic import LocalBGE,DenseIndex,reciprocal_rank_fusion


def memory():
    value=psutil.Process().memory_info()
    return {'rss_bytes':value.rss,'process_cumulative_peak_working_set_bytes':getattr(value,'peak_wset',None)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--phase-dir',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=Path('configs/pilot-v1.manifest.json'))
    parser.add_argument('--diagnostics',type=Path,default=Path('configs/pilot-v1.diagnostics.json'))
    parser.add_argument('--output',type=Path,default=Path('output/pilot-v1'))
    parser.add_argument('--max-build-seconds',type=float,default=600)
    args=parser.parse_args()
    if not 1<=args.max_build_seconds<=600: raise ValueError('Pilot build budget must be at most 600 seconds')
    args.output.mkdir(parents=True,exist_ok=True);path=args.output/'pilot.nano.json'
    if path.exists(): raise ValueError('Existing pilot index: choose a new output directory; refusing overwrite')
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    diagnostics=json.loads(args.diagnostics.read_text(encoding='utf-8'))
    reject_answer_fields(manifest);reject_answer_fields(diagnostics)
    manifest_hash=sha256_file(args.manifest);diagnostics_hash=sha256_file(args.diagnostics)
    if manifest_hash!=diagnostics['manifest_sha256'] or len(manifest['documents'])!=10: raise ValueError('Frozen manifest mismatch')
    corpus=args.audit_dir/'regions.jsonl';before=sha256_file(corpus)
    if before!=manifest['source_corpus_sha256']: raise ValueError('Source corpus mismatch')
    audit=json.loads((args.audit_dir/'data-audit.json').read_text(encoding='utf-8'))
    reviews=json.loads((args.phase_dir/'page-review.json').read_text(encoding='utf-8'))
    selected,rule=select_pilot(audit['dataset']['documents'],reviews)
    if rule!=manifest['selection_rule'] or [d['name'] for d in selected]!=[d['name'] for d in manifest['documents']]:
        raise ValueError('Frozen sampling cannot be reproduced')
    names={d['name'] for d in selected};bydoc=defaultdict(list)
    for row in load_evidence(corpus):
        if row['document_name'] in names: bydoc[row['document_name']].append(row)
    rows=[]
    for doc in manifest['documents']:
        scoped=sorted(bydoc[doc['name']],key=lambda r:r['evidence_id'])
        evidence_hash=hashlib.sha256(json.dumps(sorted(r['evidence_id'] for r in scoped)).encode()).hexdigest()
        if len(scoped)!=doc['regions'] or evidence_hash!=doc['evidence_ids_sha256'] or any(r['doc_sha256']!=doc['sha256'] for r in scoped):
            raise ValueError('Frozen document coverage mismatch')
        rows.extend(scoped);bydoc[doc['name']]=scoped
    for case in diagnostics['cases']:
        if case['document'] and case['document'] not in names: raise ValueError('Diagnostic outside pilot')
        if not set(case['source_documents'])<=names: raise ValueError('Diagnostic source reference outside pilot')
    per_document=[];cases=[];verified={};chunks=[]
    with patch.object(socket.socket,'connect',side_effect=RuntimeError('Pilot prohibits socket connects')) as network:
        started=time.perf_counter();model=LocalBGE(args.model_dir);model_load_seconds=time.perf_counter()-started
        print(f'Model ready in {model_load_seconds:.3f}s; CPU batch 1; frozen 10-document corpus',flush=True)
        build_started=time.perf_counter()
        for doc in manifest['documents']:
            doc_started=time.perf_counter();totals=defaultdict(float);doc_chunks=0
            for row in bydoc[doc['name']]:
                if time.perf_counter()-build_started>args.max_build_seconds: raise TimeoutError('Pilot source-encoding budget exhausted')
                encoded=model.region(row);chunks.extend(encoded);doc_chunks+=len(encoded)
                if len(chunks)>20000: raise ValueError('Pilot chunk budget exceeded')
                for key,value in model.last_region_stats.items(): totals[key]+=value
            item={'document_name':doc['name'],'source_regions':doc['regions'],'nonempty_regions':doc['nonempty_regions'],
                'chunks':doc_chunks,'wall_seconds':round(time.perf_counter()-doc_started,3),
                'tokenization_ms':round(totals['tokenization_ms'],3),'encoding_ms':round(totals['embedding_ms'],3),
                'region_total_ms':round(totals['total_ms'],3),'source_tokens':int(totals['source_tokens']),**memory()}
            per_document.append(item)
            (args.output/'build-checkpoint.json').write_text(json.dumps({'complete':False,'documents_done':per_document,
                'manifest_sha256':manifest_hash},indent=2)+'\n',encoding='utf-8')
            print(json.dumps(item),flush=True)
        source_encoding_seconds=time.perf_counter()-build_started
        started=time.perf_counter();index=DenseIndex(path,rows,model.identity,chunks,source_tokenizer=model.tokenizer);serialization_seconds=time.perf_counter()-started
        lexical={};lexical_build_ms={}
        for scope in sorted({case['document'] for case in diagnostics['cases']}):
            started=time.perf_counter();lexical[scope]=BM25(bydoc[scope] if scope else rows)
            lexical_build_ms[scope or 'pilot_all']=round((time.perf_counter()-started)*1000,3)
        assembler=ContextAssembler(rows)
        browser=EvidenceBrowser(args.audit_dir,args.phase_dir);browser.cache=args.output/'citation-cache';browser.cache.mkdir(exist_ok=True)
        for case in diagnostics['cases']:
            query_started=time.perf_counter();vector,qstats=model.query(case['query'])
            started=time.perf_counter();bm25=lexical[case['document']].search(case['query'],5);bm25_ms=(time.perf_counter()-started)*1000
            started=time.perf_counter();dense=index.search(vector,model.identity,5,document=case['document']);dense_ms=(time.perf_counter()-started)*1000
            started=time.perf_counter();rrf=reciprocal_rank_fusion([bm25,dense],5,matched_chunk_validator=index.validate_matched_chunk);rrf_ms=(time.perf_counter()-started)*1000
            started=time.perf_counter();contexts={name:assembler.assemble(hits) for name,hits in [('bm25',bm25),('dense',dense),('rrf',rrf)]}
            context_ms=(time.perf_counter()-started)*1000;retrieval_total_ms=(time.perf_counter()-query_started)*1000
            citation_started=time.perf_counter()
            all_rows=bm25+dense+rrf+[r for context in contexts.values() for r in context['added_regions']]
            for hit in all_rows:
                eid=hit['evidence_id']
                if eid not in verified: verified[eid]=browser.verify_citation(eid)
                if not verified[eid]['all_passed']: raise ValueError('Pilot source citation failed')
                if case['document'] and hit['document_name']!=case['document']: raise ValueError('Document scope violation')
            citation_ms=(time.perf_counter()-citation_started)*1000
            routes={'bm25':bm25,'dense':dense,'rrf':rrf}
            support={name:answer_support_state(hits,[verified[h['evidence_id']] for h in hits]) for name,hits in routes.items()}
            if any(s['supported_answer'] or s['answer_generated'] for s in support.values()): raise ValueError('No-reader support invariant failed')
            result={**case,'routes':routes,'contexts':contexts,'support':support,'query_stats':qstats,
                'timings_ms':{'bm25_lookup':round(bm25_ms,3),'dense_lookup':round(dense_ms,3),'rrf':round(rrf_ms,3),
                    'context_assembly':round(context_ms,3),'complete_retrieval':round(retrieval_total_ms,3),'citation_audit':round(citation_ms,3)},
                'source_document_presence':{name:sorted({h['document_name'] for h in hits}&set(case['source_documents'])) for name,hits in routes.items()},
                'interpretation':'Source-selected candidate/context observation only; no labeled relevance or answer metric'}
            cases.append(result)
            print(json.dumps({'case':case['id'],'retrieval_ms':result['timings_ms']['complete_retrieval'],
                'candidates':{n:len(h) for n,h in routes.items()},'supported_answer':False}),flush=True)
        report={'kind':'10-document pilot development diagnostics; not official Accuracy/Recall','complete':True,
            'manifest_sha256':manifest_hash,'diagnostics_sha256':diagnostics_hash,'pilot_documents':[d['name'] for d in selected],
            'source_regions':len(rows),'nonempty_regions':manifest['total_nonempty_regions'],'dense_chunks':index.chunk_count,
            'model_identity':model.identity,'model_load_seconds':round(model_load_seconds,3),
            'source_encoding_seconds':round(source_encoding_seconds,3),'index_serialization_seconds':round(serialization_seconds,3),
            'lexical_build_ms':lexical_build_ms,'per_document':per_document,'cases':cases,'citations':list(verified.values()),
            'all_citations_verified':all(c['all_passed'] for c in verified.values()),'socket_connect_attempts':network.call_count,
            'source_corpus_sha256':before,'source_corpus_unchanged':sha256_file(corpus)==before,'index_sha256':sha256_file(path),
            'index_bytes':path.stat().st_size,'qa_gold_read':False,'paid_api_cost':0,'supported_answer':False,'reader_present':False,
            'threshold_calibrated':False,**memory(),
            'limits':['Fixed source-derived diagnostics, not independent relevance labels','Text and original captions only; missing bodies remain missing',
                'complete_retrieval includes tokenization, encoding, three routes and context; citation audit/model loading excluded',
                'memory is Windows process working set, per-document peak is cumulative','No full100-document build or extra model']}
        if network.call_count or not report['source_corpus_unchanged']: raise ValueError('Pilot offline/source invariant failed')
        (args.output/'pilot-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (args.output/'build-checkpoint.json').write_text(json.dumps({'complete':True,'manifest_sha256':manifest_hash,
            'index_sha256':report['index_sha256'],'documents_done':per_document},indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('per_document','cases','citations')},indent=2),flush=True)

if __name__=='__main__': main()
