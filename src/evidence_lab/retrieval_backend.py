"""Actual pinned CPU retrieval backend; no labels, reader or reference-page access."""
import json
from pathlib import Path
import time
from .cli import load_evidence
from .lexical import BM25
from .retrieval_policy import CandidatePolicy
from .safe_data import sha256_file
from .semantic import LocalBGE, DenseIndex, reciprocal_rank_fusion
from .rerank import LocalReranker

class SourceRetrievalBackend:
    ROUTES=('bm25','dense','rrf','rrf_rerank','dense_rerank')
    def __init__(self,data_root,bge_dir,reranker_dir,*,threads=4):
        root=Path(data_root);build=json.loads((root/'data/full/full-build-report.json').read_text())
        corpus=root/'data/full/regions.jsonl';storage=root/'data/full/full.nano.json'
        if not build['complete'] or sha256_file(corpus)!=build['frozen_corpus_sha256'] or sha256_file(storage)!=build['index_sha256']:
            raise ValueError('Frozen corpus/index binding mismatch')
        self.rows=load_evidence(corpus);self.policy=CandidatePolicy(self.rows);self.bm25=BM25(self.policy.searchable())
        start=time.perf_counter();self.bge=LocalBGE(Path(bge_dir),threads=threads);bge_load=time.perf_counter()-start
        start=time.perf_counter()
        # This is explicitly the historical index-build identity, not a claim
        # that today's model uses the producer's Torch version.
        self.index=DenseIndex(storage,self.rows,build['model_identity'],source_tokenizer=self.bge.tokenizer)
        self.compatibility=self.index.qualify_query_runtime(self.bge)
        index_load=time.perf_counter()-start
        start=time.perf_counter();self.reranker=LocalReranker(Path(reranker_dir),threads=threads,
            matched_chunk_validator=self.index.validate_matched_chunk);reranker_load=time.perf_counter()-start
        self.identity={'actual_BGE':dict(self.bge.identity),'historical_index_build':build['model_identity'],
            'index_runtime_compatibility':self.compatibility,'actual_reranker':dict(self.reranker.identity),
            'corpus_sha256':build['frozen_corpus_sha256'],'index_sha256':build['index_sha256'],
            'device_all_retrieval_models':'cpu','BGE_load_seconds':bge_load,
            'index_load_and_qualification_seconds':index_load,'reranker_load_seconds':reranker_load,
            'ranking_pool':20,'reranker_max_pairs':40,'reference_labels_accessed':False}

    def search(self,question,route='rrf_rerank',top_k=5):
        if route not in self.ROUTES or not 1<=top_k<=20:raise ValueError('Unsupported bounded retrieval route')
        started=time.perf_counter();details={'route':route,'model_device':'cpu'}
        bm25=[];dense=[]
        if route in ('bm25','rrf','rrf_rerank'):
            t=time.perf_counter();bm25=self.bm25.search(question,20);details['BM25_seconds']=time.perf_counter()-t
        if route!='bm25':
            t=time.perf_counter();vector,qstats=self.bge.query(question)
            dense=self.index.search(vector,self.bge.identity,20,candidate_filter=self.policy.accepts)
            details.update(dense_seconds=time.perf_counter()-t,query_embedding_stats=qstats)
        if route in ('rrf','rrf_rerank'):
            t=time.perf_counter();hits=reciprocal_rank_fusion([bm25,dense],20,matched_chunk_validator=self.index.validate_matched_chunk)
            details['fusion_seconds']=time.perf_counter()-t
        else:hits=bm25 if route=='bm25' else dense
        if route in ('rrf_rerank','dense_rerank'):
            hits,notes=self.reranker.rerank(question,hits,top_k,max_pairs=40);details['reranker']=notes
        details['seconds']=time.perf_counter()-started
        return hits[:top_k],details

    def compare_routes(self,question):
        started=time.perf_counter();t=time.perf_counter();bm25=self.bm25.search(question,20);bm25_s=time.perf_counter()-t
        t=time.perf_counter();vector,qstats=self.bge.query(question)
        dense=self.index.search(vector,self.bge.identity,20,candidate_filter=self.policy.accepts);dense_s=time.perf_counter()-t
        t=time.perf_counter();rrf=reciprocal_rank_fusion([bm25,dense],20,matched_chunk_validator=self.index.validate_matched_chunk);fusion_s=time.perf_counter()-t
        fused_rerank,fused_notes=self.reranker.rerank(question,rrf,5,max_pairs=40)
        dense_rerank,dense_notes=self.reranker.rerank(question,dense,5,max_pairs=40)
        routes={'bm25':bm25[:5],'dense':dense[:5],'rrf':rrf[:5],'rrf_rerank':fused_rerank,'dense_rerank':dense_rerank}
        seconds={'bm25':bm25_s,'dense':dense_s,'rrf':bm25_s+dense_s+fusion_s,
            'rrf_rerank':bm25_s+dense_s+fusion_s+fused_notes['wall_ms']/1000,
            'dense_rerank':dense_s+dense_notes['wall_ms']/1000}
        return routes,{'route_seconds':seconds,'query_embedding_stats':qstats,'rrf_reranker':fused_notes,
            'dense_reranker':dense_notes,'all_comparison_seconds':time.perf_counter()-started,
            'timing_scope':'Warm actual route components; query/dense shared once for this comparison; no cold loads or later source checks'}
