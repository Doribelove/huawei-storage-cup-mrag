"""Source-selected development diagnostics, never QA gold or a formal benchmark."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch
import psutil
from evidence_lab.cli import load_evidence
from evidence_lab.demo import EvidenceBrowser
from evidence_lab.lexical import BM25
from evidence_lab.safe_data import sha256_file
from evidence_lab.semantic import LocalBGE,DenseIndex,reciprocal_rank_fusion

CASES=[
 {'id':'standardization_paraphrase','query':'How does standardization relate to the SDCA process?','kind':'source_paraphrase'},
 {'id':'design_standard_paraphrase','query':'Which framework helps workers design and plan a new workplace standard?','kind':'source_paraphrase'},
 {'id':'caption_exact','query':'SDCA Cycle','kind':'source_caption'},
 {'id':'nonsense_probe','query':'zzzxqvnonev290','kind':'unsupported_nonsense'},
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--phase-dir',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,default=Path('models/bge-m3'))
    parser.add_argument('--output',type=Path,default=Path('output/audit/semantic'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=args.audit_dir/'regions.jsonl'
    audit=json.loads((args.audit_dir/'data-audit.json').read_text(encoding='utf-8'))
    before=sha256_file(source)
    if before!=audit['evidence_corpus_sha256']: raise ValueError('Source corpus hash mismatch')
    rows=[r for r in load_evidence(source) if r['document_name']=='education_3501298']
    rows.sort(key=lambda r:r['evidence_id'])
    if len(rows)!=97 or len({r['doc_sha256'] for r in rows})!=1: raise ValueError('Unexpected diagnostic document scope')
    lexical=BM25(rows)
    results=[];chunks=[]
    # A blocked socket connect is counted and fails the run. Offline flags alone are not our evidence.
    with patch.object(socket.socket,'connect',side_effect=RuntimeError('Local diagnostic prohibits socket connects')) as network:
        started=time.perf_counter();model=LocalBGE(args.model_dir);load_seconds=time.perf_counter()-started
        query_vector,query_stats=model.query(CASES[0]['query'])
        print(json.dumps({'milestone':'real_query_embedded','query':CASES[0]['query'],'query_stats':query_stats,
            'dimension':len(query_vector),'norm':float((query_vector@query_vector)**0.5),'load_seconds':round(load_seconds,3)}),flush=True)
        started=time.perf_counter()
        path=args.output/'bge-m3-diagnostic.nano.json'
        reused=path.exists()
        if not reused:
            for i,row in enumerate(rows,1):
                chunks.extend(model.region(row))
                if i%10==0: print(f'Encoded {i}/{len(rows)} source regions; {len(chunks)} chunks',flush=True)
            # Nano mutates upsert dictionaries; retain independent numeric/timing metadata first.
            chunk_notes=[{k:v for k,v in c.items() if k!='__vector__'} for c in chunks]
            index=DenseIndex(path,rows,model.identity,chunks,source_tokenizer=model.tokenizer)
        else:
            chunk_notes=[];index=DenseIndex(path,rows,model.identity,source_tokenizer=model.tokenizer)
        build_seconds=time.perf_counter()-started
        browser=EvidenceBrowser(args.audit_dir,args.phase_dir)
        browser.cache=args.output/'citation-cache';browser.cache.mkdir(exist_ok=True)
        verified={}
        for number,case in enumerate(CASES):
            if number: query_vector,query_stats=model.query(case['query'])
            started=time.perf_counter();bm25=lexical.search(case['query'],5);bm25_ms=(time.perf_counter()-started)*1000
            started=time.perf_counter();dense=index.search(query_vector,model.identity,5);dense_ms=(time.perf_counter()-started)*1000
            started=time.perf_counter();fused=reciprocal_rank_fusion([bm25,dense],5,matched_chunk_validator=index.validate_matched_chunk);fusion_ms=(time.perf_counter()-started)*1000
            for hit in bm25+dense+fused:
                eid=hit['evidence_id']
                if eid not in verified: verified[eid]=browser.verify_citation(eid)
                if not verified[eid]['all_passed']: raise ValueError('Returned citation fails source checks')
            results.append({**case,'query_stats':query_stats,'query_vector_norm':float((query_vector@query_vector)**0.5),
                'routes':{'bm25':bm25,'dense':dense,'rrf':fused},
                'latency_ms':{'bm25':round(bm25_ms,3),'dense_index_query':round(dense_ms,3),'rrf':round(fusion_ms,3)},
                'answer_generated':False,'interpretation':'Candidate evidence only; dense top-k is not a calibrated answerability test'})
        memory=psutil.Process().memory_info()
        report={'kind':'four source-selected single-document development diagnostics; not competition Accuracy or Recall',
            'case_selection':'Queries derived from source page 3 and one unique nonsense probe; no queries_30 gold read',
            'document_scope':'education_3501298','source_regions':len(rows),'nonempty_regions':sum(bool(r['text'].strip()) for r in rows),
            'source_corpus_sha256':before,'model_identity':model.identity,'model_load_seconds':round(load_seconds,3),
            'index_sha256':sha256_file(path),'index_chunks':index.chunk_count,'index_reused':reused,
            'index_build_seconds':round(build_seconds,3),'chunks':chunk_notes,'cases':results,
            'citations':list(verified.values()),'all_citations_verified':all(v['all_passed'] for v in verified.values()),
            'peak_working_set_bytes':getattr(memory,'peak_wset',None),'rss_bytes':memory.rss,
            'socket_connect_attempts':network.call_count,'gold_answers_read':False,'old_vectors_reused':False,
            'source_corpus_unchanged':sha256_file(source)==before,
            'versions':{name:importlib.metadata.version(name) for name in ['torch','transformers','safetensors','nano-vectordb','numpy','psutil']},
            'limits':['Only source text and original figure captions are represented; no new vision understanding',
                'No held-out relevance labels, answer generation, calibrated refusal, GPU or official hardware timings',
                'Four manually selected cases cannot support generalization claims; warm per-case times exclude model and index loading',
                'RRF uses constant 60 without tuning; equal ranks can prefer lexical route by deterministic ID tie-break']}
        if report['socket_connect_attempts'] or not report['source_corpus_unchanged']: raise ValueError('Offline/source integrity invariant failed')
        (args.output/'semantic-diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('cases','chunks','citations')},indent=2),flush=True)


if __name__=='__main__': main()
