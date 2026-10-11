"""Pinned local BGE dense route and bounded rank fusion; no answer generation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
from weakref import WeakKeyDictionary
import numpy as np
from .safe_data import evidence_output,reject_answer_fields,sha256_file

MODEL_ID='BAAI/bge-m3'
REVISION='5617a9f61b028005a4858fdac845db406aefb181'
SOURCE_WEIGHTS_SHA256='b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38'
SAFE_WEIGHTS_SHA256='002003b14dd71c1a3df0a66cdf89b4a492bd64dcd594d856991a2e45fdf49ff1'
TOKENIZER_SHA256='21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08'

_LOADED_BGE = WeakKeyDictionary()
_QUALIFIED_QUERY_RUNTIMES = WeakKeyDictionary()

def _live_bge(model):
    state = _LOADED_BGE.get(model)
    return bool(state and model.model is state[0] and model.tokenizer is state[1]
                and json.dumps(model.identity, sort_keys=True) == state[2])


def valid_vector(vector,dimension=1024):
    vector=np.asarray(vector,dtype=np.float32)
    if vector.shape!=(dimension,) or not np.isfinite(vector).all() or np.linalg.norm(vector)==0:
        raise ValueError('Invalid dense vector dimension/value')
    return vector/np.linalg.norm(vector)


def token_windows(count:int,capacity:int,overlap:int)->list[tuple[int,int]]:
    if count<0 or capacity<1 or not 0<=overlap<capacity: raise ValueError('Invalid token window budget')
    spans=[]
    start=0
    while start<count:
        end=min(start+capacity,count);spans.append((start,end))
        if end==count: break
        start=end-overlap
    return spans


class LocalBGE:
    def __init__(self,model_dir:Path,max_length:int=512,overlap:int=64,threads:int=8):
        if not 16<=max_length<=512 or not 0<=overlap<max_length-2 or not 1<=threads<=16:
            raise ValueError('CPU prototype budget exceeded')
        for name,expected in [('model.safetensors',SAFE_WEIGHTS_SHA256),('tokenizer.json',TOKENIZER_SHA256)]:
            if sha256_file(model_dir/name)!=expected: raise ValueError(f'Pinned local file mismatch: {name}')
        for name,size,expected in [('config.json',687,'e6eda1c72da8f9dc30fdd9b69c73d35af3b7a7ad'),
                ('tokenizer_config.json',444,'dc69ac559dcba2694012009aaa108c614541789a'),
                ('special_tokens_map.json',964,'b1879d702821e753ffe4245048eee415d54a9385')]:
            raw=(model_dir/name).read_bytes()
            if len(raw)!=size or hashlib.sha1(b'blob '+str(size).encode()+b'\0'+raw).hexdigest()!=expected:
                raise ValueError(f'Pinned local configuration mismatch: {name}')
        conversion=json.loads((model_dir/'conversion-manifest.json').read_text(encoding='utf-8'))
        if conversion.get('revision')!=REVISION or conversion.get('source_weights_sha256')!=SOURCE_WEIGHTS_SHA256 or not conversion.get('all_tensor_values_shapes_dtypes_equal'):
            raise ValueError('Unverified conversion provenance')
        # Both flags and local_files_only prevent fallback downloads.
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
        import torch
        from transformers import AutoModel,AutoTokenizer
        self.torch=torch;torch.set_num_threads(threads)
        self.tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,use_fast=True)
        self.model,loading=AutoModel.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,
            use_safetensors=True,output_loading_info=True,dtype=torch.float32,attn_implementation='eager')
        if loading['missing_keys'] or loading['unexpected_keys'] or loading['mismatched_keys'] or loading['error_msgs']:
            raise ValueError(f'Model state mismatch: {loading}')
        self.model.eval();self.model.requires_grad_(False)
        if self.model.config.hidden_size!=1024: raise ValueError('Unexpected model dimension')
        self.max_length,self.overlap=max_length,overlap
        self.identity={'model_id':MODEL_ID,'revision':REVISION,'source_weights_sha256':SOURCE_WEIGHTS_SHA256,
            'loaded_safetensors_sha256':SAFE_WEIGHTS_SHA256,'tokenizer_sha256':TOKENIZER_SHA256,
            'dimension':1024,'pooling':'last_hidden_state[:,0] (CLS)','normalization':'L2',
            'device':'cpu','dtype':'float32','max_length':max_length,'overlap_tokens':overlap,
            'query_instruction':None,'trust_remote_code':False,'local_files_only':True,
            'torch_version':torch.__version__,'threads':threads,'batch_size':1,'attention':'eager'}
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError('Qualified BGE runtime must actually use CPU FP32')
        _LOADED_BGE[self] = (self.model, self.tokenizer, json.dumps(self.identity, sort_keys=True))

    def _encode_ids(self,ids:list[int])->np.ndarray:
        torch=self.torch
        if torch.get_num_threads()!=self.identity['threads']: torch.set_num_threads(self.identity['threads'])
        ids=self.tokenizer.build_inputs_with_special_tokens(ids)
        if len(ids)>self.max_length: raise ValueError('Encoding exceeds token budget; truncation is forbidden')
        inputs={'input_ids':torch.tensor([ids],dtype=torch.long),'attention_mask':torch.ones((1,len(ids)),dtype=torch.long)}
        with torch.inference_mode():
            hidden=self.model(**inputs).last_hidden_state[:,0]
            vector=torch.nn.functional.normalize(hidden,p=2,dim=1)[0].cpu().numpy()
        return valid_vector(vector)

    def query(self,text:str)->tuple[np.ndarray,dict]:
        total_started=time.perf_counter()
        if not isinstance(text,str) or not text.strip() or len(text)>500: raise ValueError('Query must be nonempty and at most 500 characters')
        tokenize_started=time.perf_counter()
        ids=self.tokenizer(text,add_special_tokens=False,truncation=False)['input_ids']
        tokenization_ms=(time.perf_counter()-tokenize_started)*1000
        if len(ids)+2>self.max_length: raise ValueError('Query exceeds token budget')
        started=time.perf_counter();vector=self._encode_ids(ids)
        return vector,{'tokens_with_specials':len(ids)+2,'truncated':False,'embedding_ms':round((time.perf_counter()-started)*1000,3),
            'tokenization_ms':round(tokenization_ms,3),'query_embedding_total_ms':round((time.perf_counter()-total_started)*1000,3)}

    def region(self,record:dict)->list[dict]:
        total_started=time.perf_counter()
        self.last_region_stats={'tokenization_ms':0.0,'embedding_ms':0.0,'total_ms':0.0,'source_tokens':0,'chunks':0}
        row=evidence_output(record)
        if row.get('role')!='source_pdf_region': raise ValueError('Only source PDF regions can enter the dense index')
        text=row['text']
        if not isinstance(text,str) or len(text)>100000: raise ValueError('Region text exceeds budget')
        if not text.strip(): return []
        tokenize_started=time.perf_counter()
        encoded=self.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,truncation=False)
        tokenization_ms=(time.perf_counter()-tokenize_started)*1000
        ids,offsets=encoded['input_ids'],encoded['offset_mapping']
        if len(ids)>20000: raise ValueError('Region token budget exceeded')
        chunks=[]
        for start,end in token_windows(len(ids),self.max_length-2,self.overlap):
            char_start=0 if start==0 else offsets[start][0]
            char_end=len(text) if end==len(ids) else offsets[end-1][1]
            started=time.perf_counter();vector=self._encode_ids(ids[start:end])
            chunks.append({'__id__':f"{row['evidence_id']}:tokens:{start}:{end}",'__vector__':vector,
                'parent_evidence_id':row['evidence_id'],'source_token_span':[start,end],'source_char_span':[char_start,char_end],
                'tokens_with_specials':end-start+2,'embedding_ms':round((time.perf_counter()-started)*1000,3)})
        self.last_region_stats={'tokenization_ms':round(tokenization_ms,3),'embedding_ms':round(sum(c['embedding_ms'] for c in chunks),3),
            'total_ms':round((time.perf_counter()-total_started)*1000,3),'source_tokens':len(ids),'chunks':len(chunks)}
        return chunks


class DenseIndex:
    def __init__(self,path:Path,records:list[dict],identity:dict,chunks:list[dict]|None=None,*,source_tokenizer=None):
        from nano_vectordb import NanoVectorDB
        self.records={r['evidence_id']:evidence_output(r) for r in records}
        if len(self.records)!=len(records): raise ValueError('Duplicate evidence IDs')
        if any(r.get('role')!='source_pdf_region' for r in self.records.values()): raise ValueError('Only source PDF regions accepted')
        self.identity=identity;self.source_tokenizer=source_tokenizer;self._source_geometry={}
        self.records_hash=hashlib.sha256(json.dumps(list(self.records.values()),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if chunks is not None:
            if path.exists(): raise ValueError('Refusing to overwrite an existing dense index')
            if not chunks or len(chunks)>20000: raise ValueError('Dense chunk budget exceeded')
            # Validate all metadata before normalization, database initialization or persistence.
            for chunk in chunks: reject_answer_fields(chunk)
            seen=set()
            for chunk in chunks:
                parent=chunk.get('parent_evidence_id')
                if parent not in self.records: raise ValueError('Invalid chunk link')
                self._check_chunk_shape({k:chunk.get(k) for k in ('__id__','source_token_span','source_char_span','tokens_with_specials')},self.records[parent]['text'])
                if chunk['__id__'] in seen or chunk['parent_evidence_id'] not in self.records: raise ValueError('Invalid chunk link')
                seen.add(chunk['__id__']);chunk['__vector__']=valid_vector(chunk['__vector__'])
        else:
            # Validate numeric storage and QA isolation before Nano parses it.
            from .nano_adapter import ReadOnlyNanoIndex
            checked=ReadOnlyNanoIndex(path)
            checked.assert_unchanged()
            self._stored_matrix = checked.matrix
            self._stored_rows = checked.records
            self._storage_path = path
            self._storage_sha256 = checked.before_hash
        self.database=NanoVectorDB(1024,storage_file=str(path))
        if chunks is not None:
            self.database.upsert(chunks)
            self.database.store_additional_data(embedding_identity=identity,records_sha256=self.records_hash)
            self.database.save()
        meta=self.database.get_additional_data()
        if meta.get('embedding_identity')!=identity or meta.get('records_sha256')!=self.records_hash:
            raise ValueError('Index/query model or evidence provenance mismatch')
        self.chunk_count=len(self.database)
        chunk_rows=chunks if chunks is not None else checked.records
        if any(item.get('parent_evidence_id') not in self.records for item in chunk_rows): raise ValueError('Invalid stored parent evidence')
        self.indexed_parents={item['parent_evidence_id'] for item in chunk_rows}
        self.indexed_documents={self.records[item['parent_evidence_id']]['document_name'] for item in chunk_rows}
        # Canonical persisted metadata is kept separate from mutable search/output dictionaries.
        self.chunk_metadata={}
        for item in chunk_rows:
            cid=item.get('__id__');parent=item['parent_evidence_id']
            if not isinstance(cid,str) or not cid or cid in self.chunk_metadata:
                raise ValueError('Duplicate/invalid stored chunk ID')
            meta={k:item.get(k) for k in ('__id__','source_token_span','source_char_span','tokens_with_specials')}
            self._check_chunk_shape(meta,self.records[parent]['text'])
            self.chunk_metadata[cid]=(parent,tuple(meta['source_token_span']),tuple(meta['source_char_span']),meta['tokens_with_specials'])

    @staticmethod
    def _check_chunk_shape(meta:dict,text:str):
        if not isinstance(meta,dict) or set(meta)!={'__id__','source_token_span','source_char_span','tokens_with_specials'} or not isinstance(meta['__id__'],str) or not meta['__id__']:
            raise ValueError('Invalid matched chunk schema')
        for field,limit in [('source_token_span',20000),('source_char_span',len(text))]:
            span=meta[field]
            if not isinstance(span,(list,tuple)) or len(span)!=2 or any(isinstance(v,bool) or not isinstance(v,int) for v in span) or not 0<=span[0]<span[1]<=limit:
                raise ValueError('Invalid original matched chunk span')
        size=meta['tokens_with_specials']
        if isinstance(size,bool) or not isinstance(size,int) or size!=meta['source_token_span'][1]-meta['source_token_span'][0]+2 or size>512:
            raise ValueError('Invalid matched chunk token count')

    def validate_matched_chunk(self,hit:dict)->dict:
        """Verify supplied match against this bound index AND original BGE token offsets.

        A plain external span or a caller-provided 'validated' flag is never trusted.
        Runtime supplies the pinned LocalBGE tokenizer already checked on load.
        """
        row=evidence_output({k:v for k,v in hit.items() if k not in ('score','route','route_ranks','matched_chunk')})
        parent=row.get('evidence_id');canonical=self.records.get(parent)
        if canonical is None or row!=canonical: raise ValueError('Matched chunk/source evidence binding mismatch')
        meta=hit.get('matched_chunk');self._check_chunk_shape(meta,row['text'])
        expected=self.chunk_metadata.get(meta['__id__'])
        supplied=(parent,tuple(meta['source_token_span']),tuple(meta['source_char_span']),meta['tokens_with_specials'])
        if expected!=supplied: raise ValueError('Matched chunk differs from bound dense index')
        if self.source_tokenizer is None: raise ValueError('Original source tokenizer is required to validate matched chunks')
        if parent not in self._source_geometry:
            encoded=self.source_tokenizer(row['text'],add_special_tokens=False,truncation=False,return_offsets_mapping=True)
            ids,offsets=encoded['input_ids'],encoded['offset_mapping']
            if len(ids)!=len(offsets) or len(ids)>20000: raise ValueError('Original source token geometry budget mismatch')
            capacity=self.identity['max_length']-2;overlap=self.identity['overlap_tokens']
            geometry={}
            for start,end in token_windows(len(ids),capacity,overlap):
                char_start=0 if start==0 else offsets[start][0]
                char_end=len(row['text']) if end==len(ids) else offsets[end-1][1]
                geometry[(start,end)]=(char_start,char_end)
            self._source_geometry[parent]=geometry
        if self._source_geometry[parent].get(expected[1])!=expected[2]:
            raise ValueError('Stored dense chunk is not an original tokenizer window')
        return {'__id__':meta['__id__'],'source_token_span':list(expected[1]),
                'source_char_span':list(expected[2]),'tokens_with_specials':expected[3]}

    def qualify_query_runtime(self, model, sample_count: int = 8) -> dict:
        """Re-encode original chunks before accepting a different observed CPU runtime.

        Stored build identity stays unchanged. Only torch version and thread count
        may differ; every embedding/preprocessing/weight/precision field must match.
        A saved report or caller-provided pass flag never authorizes the route.
        """
        if not isinstance(model, LocalBGE) or not _live_bge(model):
            raise ValueError('Runtime qualification requires genuinely loaded pinned BGE')
        if not 3 <= sample_count <= 12 or not hasattr(self, '_stored_matrix'):
            raise ValueError('Qualification requires an existing bounded source index')
        differences = {key: {'index': self.identity.get(key), 'query': model.identity.get(key)}
                       for key in set(self.identity) | set(model.identity)
                       if self.identity.get(key) != model.identity.get(key)}
        if set(differences) - {'torch_version', 'threads'}:
            raise ValueError('Index/query embedding recipe differs: '+str(sorted(differences)))
        if sha256_file(self._storage_path) != self._storage_sha256:
            raise ValueError('Index bytes changed before runtime qualification')
        norms = np.linalg.norm(self._stored_matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4, rtol=0):
            raise ValueError('Existing index is not L2-normalized')
        count = len(self._stored_rows)
        positions = sorted({round(i*(count-1)/(sample_count-1)) for i in range(sample_count)})
        samples = []; started = time.perf_counter()
        for position in positions:
            chunk = self._stored_rows[position]
            row = self.records[chunk['parent_evidence_id']]
            self.validate_matched_chunk({**row, 'matched_chunk': {key: chunk[key] for key in
                ('__id__', 'source_token_span', 'source_char_span', 'tokens_with_specials')}})
            ids = model.tokenizer(row['text'], add_special_tokens=False, truncation=False)['input_ids']
            begin, end = chunk['source_token_span']
            current = model._encode_ids(ids[begin:end]); old = self._stored_matrix[position]
            cosine = float(np.dot(current, old)/(np.linalg.norm(current)*np.linalg.norm(old)))
            maximum_error = float(np.max(np.abs(current-old)))
            if cosine < .99999 or maximum_error > 1e-4:
                raise ValueError('Re-encoded source chunk is numerically incompatible')
            samples.append({'chunk_id': chunk['__id__'], 'tokens': end-begin,
                            'cosine': cosine, 'maximum_absolute_difference': maximum_error})
        if not _live_bge(model): raise ValueError('BGE identity changed during qualification')
        report = {'schema': 'live-BGE-index-runtime-compatibility-v1', 'passed': True,
                  'index_sha256': self._storage_sha256, 'stored_build_identity': dict(self.identity),
                  'actual_query_runtime_identity': dict(model.identity), 'observed_runtime_differences': differences,
                  'samples': samples, 'dimension': 1024, 'normalization': 'L2',
                  'seconds': time.perf_counter()-started, 'index_rebuilt': False,
                  'scope': 'Fixed recipe plus representative numerical checks; not all chunks re-encoded'}
        _QUALIFIED_QUERY_RUNTIMES[self] = (model, dict(model.identity))
        return report

    def search(self,vector,query_identity:dict,top_k:int=5,document:str='',candidate_filter=None)->list[dict]:
        if query_identity!=self.identity:
            qualified = _QUALIFIED_QUERY_RUNTIMES.get(self)
            if not (qualified and _live_bge(qualified[0]) and query_identity == qualified[1]):
                raise ValueError('Index/query embedding identity mismatch; live numerical qualification required')
        if not 1<=top_k<=100: raise ValueError('top_k must be 1..100')
        # Rank every bounded prototype chunk, then deduplicate regions; avoid short top-k losing parents.
        if document and document not in {r['document_name'] for r in self.records.values()}: raise ValueError('Unknown dense document scope')
        query=valid_vector(vector)
        eligible={eid for eid in self.indexed_parents
                  if (not document or self.records[eid]['document_name']==document)
                  and (candidate_filter is None or candidate_filter(self.records[eid]))}
        if not eligible: return []  # Nano0.0.4.3 raises on an empty filter's NumPy dtype.
        scope=(lambda item:item['parent_evidence_id'] in eligible) if document or candidate_filter else None
        if document and document not in self.indexed_documents: return []
        hits=self.database.query(query,top_k=self.chunk_count,filter_lambda=scope)
        hits.sort(key=lambda h:(-float(h['__metrics__']),h['__id__']))
        selected=[];seen=set()
        for hit in hits:
            parent=hit['parent_evidence_id']
            if parent in seen: continue
            seen.add(parent)
            result={'route':'bge_dense','score':round(float(hit['__metrics__']),6),**self.records[parent],
                'matched_chunk':{k:hit[k] for k in ('__id__','source_token_span','source_char_span','tokens_with_specials')}}
            if self.source_tokenizer is not None: result['matched_chunk']=self.validate_matched_chunk(result)
            selected.append(result)
            if len(selected)==top_k: break
        return selected


def reciprocal_rank_fusion(routes:list[list[dict]],top_k:int=5,rank_constant:int=60,*,matched_chunk_validator=None)->list[dict]:
    if not 1<=top_k<=100 or rank_constant<1 or len(routes)>4 or any(len(route)>100 for route in routes):
        raise ValueError('Fusion budget exceeded')
    combined={}
    for route in routes:
        seen=set()
        for rank,hit in enumerate(route,1):
            # Route scores are not evidence fields; reconstruct only the source whitelist.
            row={k:v for k,v in hit.items() if k not in ('score','route','matched_chunk')}
            row=evidence_output(row);key=row['evidence_id']
            match=None
            if 'matched_chunk' in hit:
                if matched_chunk_validator is None: raise ValueError('Unverified matched chunk cannot enter fusion')
                match=matched_chunk_validator(hit)
            if key in seen: continue
            seen.add(key)
            item=combined.setdefault(key,{'row':row,'value':0.0,'ranks':[],'match':None,'match_order':None})
            if item['row']!=row: raise ValueError('Conflicting evidence between routes')
            item['value']+=1/(rank_constant+rank);item['ranks'].append({'route':hit['route'],'rank':rank})
            # Multiple valid route/query matches: retain best input rank, then canonical ID.
            # No ranking-score change; the chosen original window only guides later reranking.
            if match is not None and (item['match_order'] is None or (rank,match['__id__'])<item['match_order']):
                item['match']=match;item['match_order']=(rank,match['__id__'])
    ordered=sorted(combined.values(),key=lambda item:(-item['value'],item['row']['evidence_id']))
    return [{'route':'rrf','score':round(item['value'],8),**item['row'],'route_ranks':item['ranks'],
             **({'matched_chunk':item['match']} if item['match'] is not None else {})} for item in ordered[:top_k]]
