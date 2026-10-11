"""Two real-source development probes with genuine BGE and cross-encoder inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch
import psutil
from evidence_lab.agent import IterativeRetriever,RetrievalBudget
from evidence_lab.cli import load_evidence
from evidence_lab.demo import EvidenceBrowser
from evidence_lab.lexical import BM25
from evidence_lab.native_pdf import SourceVerifier
from evidence_lab.rerank import LocalReranker
from evidence_lab.retrieval_policy import CandidatePolicy
from evidence_lab.safe_data import sha256_file
from evidence_lab.semantic import LocalBGE,DenseIndex,reciprocal_rank_fusion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--phase-dir',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--reranker-dir',type=Path,default=Path('models/bge-reranker-v2-m3'))
    parser.add_argument('--full-freeze-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    target=args.output/'agent-reranker-report.json'
    if target.exists(): raise ValueError('New diagnostic directory required')
    manifest=json.loads(Path('configs/pilot-v1.manifest.json').read_text(encoding='utf-8'))
    original=load_evidence(args.audit_dir/'regions.jsonl');rows=[]
    for doc in manifest['documents']:
        rows.extend(sorted([r for r in original if r['document_name']==doc['name']],key=lambda r:r['evidence_id']))
    names={doc['name'] for doc in manifest['documents']}
    native=[r for r in load_evidence(args.full_freeze_dir/'native-evidence.jsonl') if r['document_name'] in names]
    sources=rows+native;policy=CandidatePolicy(sources)
    old_index=Path('output/pilot-v1/pilot.nano.json');before=sha256_file(old_index)
    corpus_sha=sha256_file(args.audit_dir/'regions.jsonl')
    with patch.object(socket.socket,'connect',side_effect=RuntimeError('Local agent diagnostic prohibits network')) as network:
        started=time.perf_counter();model=LocalBGE(args.model_dir);bge_load=time.perf_counter()-started
        index=DenseIndex(old_index,rows,model.identity,source_tokenizer=model.tokenizer)
        started=time.perf_counter();reranker=LocalReranker(args.reranker_dir,matched_chunk_validator=index.validate_matched_chunk);reranker_load=time.perf_counter()-started
        print(json.dumps({'stage':'genuine_models_loaded','bge_load_seconds':round(bge_load,3),
                          'reranker_load_seconds':round(reranker_load,3)}),flush=True)
        lexical={};browser=EvidenceBrowser(args.audit_dir,args.phase_dir)
        browser.cache=args.output/'citation-cache';browser.cache.mkdir(exist_ok=True)
        verifier=SourceVerifier(browser,native);query_stats=[]
        def retrieve(query,document,k):
            vector,stats=model.query(query);stats['actual_torch_threads_after_bge']=model.torch.get_num_threads();query_stats.append(stats)
            if document not in lexical:
                lexical[document]=BM25([r for r in policy.searchable() if not document or r['document_name']==document])
            bm25=lexical[document].search(query,k)
            dense=index.search(vector,model.identity,k,document=document,candidate_filter=policy.accepts)
            return reciprocal_rank_fusion([bm25,dense],k,matched_chunk_validator=index.validate_matched_chunk)
        agent=IterativeRetriever(sources,policy,retrieve,verifier.verify,reranker,RetrievalBudget(max_seconds=90),matched_chunk_validator=index.validate_matched_chunk)
        results=[]
        for query,scope in [('What is the responsibility of the Receiving Clerk?','crm_4502338'),
                            ('COVID-19 financial results and digitalization','')]:
            result=agent.run(query,scope);results.append(result)
            print(json.dumps({'query':query,'queries_executed':result['queries_executed'],
                              'top_source_pages':[(h['document_name'],h['page']) for h in result['hits']],
                              'wall_ms':result['wall_ms'],'pairs':result['rerank_stats']['pairs_encoded'],
                              'supported_answer':False,'source_provenance_passed':all(c['all_passed'] for c in result['source_binding_checks'])}),flush=True)
        memory=psutil.Process().memory_info()
        report={'kind':'Two real-source10-PDF development probes; not provided30 or answer Accuracy',
                'bge_identity':model.identity,'reranker_identity':reranker.identity,
                'bge_load_seconds':round(bge_load,3),'reranker_load_seconds':round(reranker_load,3),
                'results':results,'query_embedding_statistics':query_stats,'socket_connect_attempts':network.call_count,
                'qa_questions_or_answers_read':False,'original_corpus_unchanged':sha256_file(args.audit_dir/'regions.jsonl')==corpus_sha,
                'reviewed_pilot_index_unchanged':sha256_file(old_index)==before,
                'peak_process_working_set_bytes':getattr(memory,'peak_wset',None),
                'paid_api_cost':0,'answer_accuracy_measured':False,
                'limitations':['Native overlay is lexical/page evidence; old pilot dense index stays immutable',
                               'No Qwen reader has run; deterministic planning is not LLM planning',
                               'Timing recorded while a separate full BGE build competes for CPU; not formal latency']}
        if network.call_count or not report['original_corpus_unchanged'] or not report['reviewed_pilot_index_unchanged']:
            raise ValueError('Agent diagnostic offline/source/index invariant failed')
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('results','bge_identity','reranker_identity','query_embedding_statistics')},indent=2),flush=True)


if __name__=='__main__': main()
