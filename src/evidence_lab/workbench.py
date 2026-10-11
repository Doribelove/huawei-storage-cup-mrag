"""Loopback-only genuine full-corpus retrieval workbench. No reader fallback or QA labels."""
from __future__ import annotations
import argparse
from http.server import HTTPServer
import json
from pathlib import Path
import time
from .agent import IterativeRetriever,RetrievalBudget
from .cli import load_evidence
from .demo import EvidenceBrowser,make_handler
from .lexical import BM25
from .native_pdf import SourceVerifier
from .rerank import LocalReranker
from .retrieval_policy import CandidatePolicy
from .safe_data import sha256_file
from .semantic import LocalBGE,DenseIndex,reciprocal_rank_fusion


class Workbench:
    def __init__(self,audit_dir:Path,phase_dir:Path,freeze_dir:Path,model_dir:Path,reranker_dir:Path,*,data_archive:Path|None=None,cache_dir:Path|None=None):
        build=json.loads((freeze_dir/'full-build-report.json').read_text(encoding='utf-8'))
        source=freeze_dir/'regions.jsonl';index_path=freeze_dir/'full.nano.json'
        if not build['complete'] or sha256_file(source)!=build['frozen_corpus_sha256'] or sha256_file(index_path)!=build['index_sha256']:
            raise ValueError('Completed full source/index binding mismatch')
        self.build=build;self.original=EvidenceBrowser(audit_dir,phase_dir,data_archive=data_archive,cache_dir=cache_dir)
        self.original.cache=Path(cache_dir) if cache_dir is not None else freeze_dir/'workbench-page-cache'
        self.original.cache.mkdir(parents=True,exist_ok=True)
        self.sources=load_evidence(source);self.policy=CandidatePolicy(self.sources)
        self.native=[r for r in self.sources if r.get('extraction')]
        self.verifier=SourceVerifier(self.original,self.native);self.certificates={}
        self.model=LocalBGE(model_dir);self.index=DenseIndex(index_path,self.sources,self.model.identity,source_tokenizer=self.model.tokenizer)
        self.reranker=LocalReranker(reranker_dir,matched_chunk_validator=self.index.validate_matched_chunk)
        self.lexical={};self.calls={'bge_embeddings':0,'reranker_pairs':0}
        self.assets=Path(__file__).parent/'workbench_web'
        self.reviews=self.original.reviews;self.integrity=self.original.integrity
        self.agent=IterativeRetriever(self.sources,self.policy,self.hybrid,self.verify_citation,self.reranker,RetrievalBudget(),matched_chunk_validator=self.index.validate_matched_chunk)

    def metadata(self):
        meta=self.original.metadata()
        return {**meta,'regions':len(self.sources),'dense_chunks':self.index.chunk_count,'native_tables':len(self.native),
                'index_sha256':self.build['index_sha256'],'model_calls':sum(self.calls.values()),'model_call_details':self.calls.copy(),
                'modes':['bm25','dense','hybrid','rerank','agent'],'reader_present':False,'answer_generated':False,
                'boundary':'全量100PDF真实检索 · BGE与重排已运行 · Qwen阅读器未运行，当前只展示候选证据',
                'bge_identity':self.model.identity,'reranker_identity':self.reranker.identity}

    def lexical_index(self,document):
        if document and document not in self.original.documents: raise ValueError('Unknown source document')
        if document not in self.lexical:
            self.lexical[document]=BM25([r for r in self.policy.searchable() if not document or r['document_name']==document])
        return self.lexical[document]

    def hybrid(self,query,document,k):
        bm25=self.lexical_index(document).search(query,k)
        vector,stats=self.model.query(query);self.calls['bge_embeddings']+=1
        dense=self.index.search(vector,self.model.identity,k,document=document,candidate_filter=self.policy.accepts)
        return reciprocal_rank_fusion([bm25,dense],k,matched_chunk_validator=self.index.validate_matched_chunk)

    def verify_citation(self,evidence_id):
        self.certificates[evidence_id]=self.verifier.verify(evidence_id)
        return self.certificates[evidence_id]

    def search_with_options(self,query,document='',top_k=5,mode='hybrid'):
        if not isinstance(query,str) or not query.strip() or len(query)>500 or not 1<=top_k<=5 or mode not in ('bm25','dense','hybrid','rerank','agent'):
            raise ValueError('Invalid bounded workbench query/mode')
        started=time.perf_counter();before=self.calls.copy();trace=None;context=None
        if mode=='bm25': hits=self.lexical_index(document).search(query,top_k)
        elif mode=='dense':
            self.lexical_index(document)
            vector,stats=self.model.query(query);self.calls['bge_embeddings']+=1
            hits=self.index.search(vector,self.model.identity,top_k,document=document,candidate_filter=self.policy.accepts)
        elif mode in ('hybrid','rerank'):
            hits=self.hybrid(query,document,20)
            if mode=='rerank':
                hits,trace=self.reranker.rerank(query,hits,top_k)
                self.calls['reranker_pairs']+=trace['pairs_encoded']
            hits=hits[:top_k]
        else:
            trace=self.agent.run(query,document,final_regions=top_k);hits=trace['hits'];context=trace['context']
            if trace['rerank_stats']: self.calls['reranker_pairs']+=trace['rerank_stats']['pairs_encoded']
        for hit in hits:
            if not self.verify_citation(hit['evidence_id'])['all_passed']: raise ValueError('Workbench source citation failed')
            hit['anomaly_classes']=[r['visual_classification'] for r in self.original.anomalies.get((hit['document_name'],hit['page']),[])]
        delta={key:self.calls[key]-before[key] for key in before}
        return {'query':query,'document_filter':document,'mode':mode,'hits':hits,'context':context,'trace':trace,
                'latency_ms':round((time.perf_counter()-started)*1000,3),'model_calls':sum(delta.values()),
                'model_call_details':delta,'supported_answer':False,'answer_generated':False,'support_state':'requires_reader',
                'reader_status':'Qwen阅读器尚未运行；来源核验不等于语义正确性。'}

    def page_info(self,name,page):
        value=self.original.page_info(name,page)
        for row in self.native:
            if row['document_name']==name and row['page']==page:
                value['regions'].append({key:row[key] for key in ('evidence_id','type','source_bbox','bbox_normalized','text')})
        value['geometry_note']='MinerU归一化坐标或原PDF原生表格坐标，均保留独立来源；无内容猜测。'
        return value

    def rendered_page(self,name,page): return self.original.rendered_page(name,page)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--phase-dir',type=Path,required=True)
    parser.add_argument('--freeze-dir',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--reranker-dir',type=Path,default=Path('models/bge-reranker-v2-m3'))
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--data-archive',type=Path,help='Relocated original data archive; frozen SHA must match')
    parser.add_argument('--cache-dir',type=Path,help='Separate task-owned page cache')
    args=parser.parse_args()
    browser=Workbench(args.audit_dir,args.phase_dir,args.freeze_dir,args.model_dir,args.reranker_dir,data_archive=args.data_archive,cache_dir=args.cache_dir)
    server=HTTPServer(('127.0.0.1',args.port),make_handler(browser))
    print(f'Genuine full-corpus workbench: http://127.0.0.1:{args.port}; Qwen unavailable; Ctrl+C stops',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__=='__main__': main()
