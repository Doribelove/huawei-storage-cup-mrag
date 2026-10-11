"""Question-only full-corpus retrieval and source binding. Scorer labels are never read."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import socket
import statistics
import time
from unittest.mock import patch
import psutil
from evidence_lab.cli import load_evidence
from evidence_lab.demo import EvidenceBrowser
from evidence_lab.evaluation import load_questions
from evidence_lab.lexical import BM25
from evidence_lab.native_pdf import SourceVerifier
from evidence_lab.pilot import ContextAssembler
from evidence_lab.rerank import LocalReranker
from evidence_lab.retrieval_policy import CandidatePolicy
from evidence_lab.safe_data import sha256_file
from evidence_lab.semantic import DenseIndex,LocalBGE,reciprocal_rank_fusion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-dir',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--phase-dir',type=Path,required=True)
    parser.add_argument('--questions',type=Path,required=True)
    parser.add_argument('--questions-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reranker-dir',type=Path,default=Path('models/bge-reranker-v2-m3'))
    parser.add_argument('--lineage-control',action='store_true',help='Also genuinely rerank the identical pool with dense matches removed; never reuse old logits')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    target=args.output/'retrieval-predictions.json'
    if target.exists(): raise ValueError('Refusing overwrite; use a new retrieval run directory')
    if sha256_file(args.questions)!=args.questions_sha256: raise ValueError('Frozen runtime question projection mismatch')
    questions=load_questions(args.questions)
    build=json.loads((args.freeze_dir/'full-build-report.json').read_text(encoding='utf-8'))
    source=args.freeze_dir/'regions.jsonl';index_path=args.freeze_dir/'full.nano.json'
    if not build['complete'] or sha256_file(source)!=build['frozen_corpus_sha256'] or sha256_file(index_path)!=build['index_sha256']:
        raise ValueError('Completed full source/index binding mismatch')
    rows=load_evidence(source);policy=CandidatePolicy(rows)
    original=[r for r in rows if not r.get('extraction')]
    native=[r for r in rows if r.get('extraction')]
    with patch.object(socket.socket,'connect',side_effect=RuntimeError('Retrieval prohibits network')) as network:
        started=time.perf_counter();model=LocalBGE(args.model_dir);model_load=time.perf_counter()-started
        started=time.perf_counter();index=DenseIndex(index_path,rows,model.identity,source_tokenizer=model.tokenizer);index_load=time.perf_counter()-started
        started=time.perf_counter();reranker=LocalReranker(args.reranker_dir,matched_chunk_validator=index.validate_matched_chunk);reranker_load=time.perf_counter()-started
        lexical=BM25(policy.searchable());lexical_raw=BM25(original)
        lexical_no_native=BM25([r for r in policy.searchable() if not r.get('extraction')])
        assembler=ContextAssembler(rows);browser=EvidenceBrowser(args.audit_dir,args.phase_dir)
        browser.cache=args.output/'citation-cache';browser.cache.mkdir(exist_ok=True)
        verifier=SourceVerifier(browser,native);verified={};cases=[]
        for question in questions:
            started=time.perf_counter();vector,qstats=model.query(question['question'])
            bm25=lexical.search(question['question'],20)
            dense=index.search(vector,model.identity,20,candidate_filter=policy.accepts)
            rrf=reciprocal_rank_fusion([bm25,dense],20,matched_chunk_validator=index.validate_matched_chunk)
            context=assembler.assemble(rrf[:5],candidate_filter=policy.accepts)
            complete_main_ms=(time.perf_counter()-started)*1000
            reranked,rerank_stats=reranker.rerank(question['question'],rrf,5)
            main_plus_reranking_ms=(time.perf_counter()-started)*1000
            control=None;control_stats=None
            if args.lineage_control:
                control_pool=[{k:v for k,v in hit.items() if k!='matched_chunk'} for hit in rrf]
                control,control_stats=reranker.rerank(question['question'],control_pool,5)
            ablation_started=time.perf_counter()
            raw=lexical_raw.search(question['question'],20)
            no_native=lexical_no_native.search(question['question'],20)
            dense_no_native=index.search(vector,model.identity,20,candidate_filter=lambda r:policy.accepts(r) and not r.get('extraction'))
            rrf_no_native=reciprocal_rank_fusion([no_native,dense_no_native],20,matched_chunk_validator=index.validate_matched_chunk)
            ablation_ms=(time.perf_counter()-ablation_started)*1000
            routes={'bm25_raw':raw[:5],'bm25':bm25[:5],'dense':dense[:5],'rrf':rrf[:5],
                    'bm25_no_native':no_native[:5],'rrf_no_native':rrf_no_native[:5],'rrf_rerank':reranked}
            if control is not None: routes['rrf_rerank_no_dense_lineage']=control
            audit_started=time.perf_counter()
            for hits in [*routes.values(),context['added_regions']]:
                for hit in hits:
                    eid=hit['evidence_id']
                    if eid not in verified: verified[eid]=verifier.verify(eid)
                    if not verified[eid]['all_passed']: raise ValueError('Full retrieval source citation failed')
            audit_ms=(time.perf_counter()-audit_started)*1000
            item={**question,'routes':routes,'context':context,'query_stats':qstats,
                  'rerank_stats':rerank_stats,
                  'lineage_control_stats':control_stats,
                  'timings_ms':{'main_complete_retrieval':round(complete_main_ms,3),'ablations':round(ablation_ms,3),
                                'reranking':rerank_stats['wall_ms'],
                                'main_plus_reranking':round(main_plus_reranking_ms,3),
                                'separate_citation_audit':round(audit_ms,3)},
                  'supported_answer':False,'answer_generated':False,'support_state':'not_established'}
            cases.append(item)
            print(json.dumps({'sample_id':question['sample_id'],'main_retrieval_ms':round(complete_main_ms,3),
                              'citation_audit_ms':round(audit_ms,3),'rerank_ms':rerank_stats['wall_ms'],
                              'lineage_control_ms':control_stats['wall_ms'] if control_stats else None,'answer_generated':False}),flush=True)
            (args.output/'retrieval-checkpoint.json').write_text(json.dumps({'complete':False,'questions_done':len(cases)},indent=2)+'\n',encoding='utf-8')
        memory=psutil.Process().memory_info()
        report={'complete':True,'kind':'Provided question-only full100-PDF genuine retrieval; no answer accuracy',
                'questions_sha256':args.questions_sha256,'frozen_corpus_sha256':build['frozen_corpus_sha256'],
                'index_sha256':build['index_sha256'],'model_identity':model.identity,'model_load_seconds':round(model_load,3),
                'index_load_seconds':round(index_load,3),'question_count':len(cases),'cases':cases,'citations':list(verified.values()),
                'reranker_identity':reranker.identity,'reranker_load_seconds':round(reranker_load,3),
                'lineage_control_enabled':args.lineage_control,
                'lineage_control_median_ms':statistics.median(c['lineage_control_stats']['wall_ms'] for c in cases) if args.lineage_control else None,
                'reranking_median_ms':statistics.median(c['timings_ms']['reranking'] for c in cases),
                'main_plus_reranking_median_ms':statistics.median(c['timings_ms']['main_plus_reranking'] for c in cases),
                'all_citations_verified':True,'main_complete_retrieval_median_ms':statistics.median(c['timings_ms']['main_complete_retrieval'] for c in cases),
                'main_complete_retrieval_max_ms':max(c['timings_ms']['main_complete_retrieval'] for c in cases),
                'socket_connect_attempts':network.call_count,'paid_api_cost':0,'qa_answers_or_reference_labels_read':False,
                'source_corpus_unchanged':sha256_file(source)==build['frozen_corpus_sha256'],
                'index_unchanged':sha256_file(index_path)==build['index_sha256'],
                'peak_process_working_set_bytes':getattr(memory,'peak_wset',None),
                'protocol':{'main':'BM25+DENSE20-region pools; RRF constant60; first5regions, bounded context',
                            'ablations':['Original BM25 without source noise policy/native overlay','Native overlay disabled in both retrieval routes','Pinned real cross-encoder reranks20RRFcandidates'],
                            'timing':'Query tokenization+embedding+main two routes+fusion+context; excludes cold model/index load, independent ablations and citation audit'},
                'supported_answer':False,'answer_generated':False}
        if network.call_count or not report['source_corpus_unchanged'] or not report['index_unchanged']:
            raise ValueError('Full retrieval offline/source/index invariant failed')
        if args.lineage_control:
            report['protocol']['lineage_control']='Fresh second inference on the identical20RRFpool with only matched_chunk removed; fixed route runs first. Control time excluded from main_plus_reranking.'
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (args.output/'retrieval-checkpoint.json').write_text(json.dumps({'complete':True,'questions_done':len(cases)},indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('cases','citations','model_identity')},indent=2),flush=True)


if __name__=='__main__': main()
