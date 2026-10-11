"""Pinned local cross-encoder; scores rank source regions, never establish answer support."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import time
from .lexical import tokens
from .safe_data import evidence_output,sha256_file
from .semantic import token_windows

MODEL_ID='BAAI/bge-reranker-v2-m3'
REVISION='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
PINNED={
    'config.json':(795,'git_blob_sha1','9f62673cb00ec41dcec8947b9ed16f6f2eb23ba2'),
    'model.safetensors':(2271071852,'sha256','d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286'),
    'special_tokens_map.json':(964,'git_blob_sha1','b1879d702821e753ffe4245048eee415d54a9385'),
    'tokenizer.json':(17098273,'sha256','69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15'),
    'tokenizer_config.json':(1173,'git_blob_sha1','328a00a9a560aadcf2a3064f917517359eb3cc26')}


def select_passage_windows(text: str, offsets: list, capacity: int, query: str,
                           matched_char_span=None, max_windows: int = 2) -> tuple[list[dict],int]:
    """Select bounded original spans without replacing omitted text with generated summaries."""
    if not 1<=max_windows<=2: raise ValueError('Passage window budget exceeded')
    spans=token_windows(len(offsets),capacity,min(32,capacity-1));terms=set(tokens(query));windows=[]
    for start,end in spans:
        char_start=0 if start==0 else offsets[start][0]
        char_end=len(text) if end==len(offsets) else offsets[end-1][1]
        lexical_overlap=len(terms & set(tokens(text[char_start:char_end])))
        dense_overlap=0
        if matched_char_span is not None:
            if len(matched_char_span)!=2 or not all(isinstance(v,int) and not isinstance(v,bool) for v in matched_char_span) or not 0<=matched_char_span[0]<matched_char_span[1]<=len(text):
                raise ValueError('Invalid original matched chunk span')
            dense_overlap=max(0,min(char_end,matched_char_span[1])-max(char_start,matched_char_span[0]))
        windows.append({'source_token_span':[start,end],'source_char_span':[char_start,char_end],
                        'lexical_overlap':lexical_overlap,'dense_overlap':dense_overlap})
    selected=sorted(windows,key=lambda w:(-w['dense_overlap'],-w['lexical_overlap'],w['source_token_span'][0]))[:max_windows]
    return selected,len(windows)-len(selected)


class LocalReranker:
    def __init__(self,model_dir:Path,threads:int=4,max_length:int=512,*,matched_chunk_validator=None):
        if not 1<=threads<=8 or not 64<=max_length<=512: raise ValueError('Reranker CPU budget exceeded')
        for name,(size,kind,expected) in PINNED.items():
            path=model_dir/name
            if path.stat().st_size!=size: raise ValueError('Pinned reranker file size mismatch')
            actual=sha256_file(path) if kind=='sha256' else hashlib.sha1(b'blob '+str(size).encode()+b'\0'+path.read_bytes()).hexdigest()
            if actual!=expected: raise ValueError('Pinned reranker file hash mismatch')
        manifest=json.loads((model_dir/'source-manifest.json').read_text(encoding='utf-8'))
        if manifest['revision']!=REVISION or manifest['model_id']!=MODEL_ID: raise ValueError('Reranker provenance mismatch')
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
        os.environ['TOKENIZERS_PARALLELISM']='false'
        import torch
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        self.torch=torch;torch.set_num_threads(threads)
        self.tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,use_fast=True)
        self.model,loading=AutoModelForSequenceClassification.from_pretrained(model_dir,local_files_only=True,
            trust_remote_code=False,use_safetensors=True,dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
        if any(loading[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Reranker model state mismatch')
        if self.model.config.num_labels!=1: raise ValueError('Unexpected reranker output shape')
        self.model.eval();self.model.requires_grad_(False);self.max_length=max_length
        self.matched_chunk_validator=matched_chunk_validator
        self.identity={'model_id':MODEL_ID,'revision':REVISION,'weights_sha256':PINNED['model.safetensors'][2],
                       'tokenizer_sha256':PINNED['tokenizer.json'][2],'device':'cpu','dtype':'float32','threads':threads,
                       'max_length':max_length,'batch_size':1,'passage_max_windows':2,'aggregation':'max raw logit',
                       'trust_remote_code':False,'local_files_only':True,'torch_version':torch.__version__}

    def rerank(self,query:str,hits:list[dict],top_k:int=5,max_pairs:int=40)->tuple[list[dict],dict]:
        started=time.perf_counter()
        if self.torch.get_num_threads()!=self.identity['threads']: self.torch.set_num_threads(self.identity['threads'])
        if not isinstance(query,str) or not query.strip() or len(query)>500 or not 1<=top_k<=20 or not 1<=max_pairs<=40 or len(hits)>20:
            raise ValueError('Reranking query/candidate/pair budget exceeded')
        qids=self.tokenizer(query,add_special_tokens=False,truncation=False)['input_ids']
        capacity=self.max_length-len(qids)-self.tokenizer.num_special_tokens_to_add(pair=True)
        if capacity<32: raise ValueError('Query leaves insufficient passage context; no implicit truncation')
        result=[];pairs=0;notes=[];seen=set()
        for hit in hits:
            row=evidence_output({k:v for k,v in hit.items() if k not in ('score','route','route_ranks')})
            if row.get('role')!='source_pdf_region' or row['evidence_id'] in seen:
                raise ValueError('Reranker requires unique original-source candidates')
            seen.add(row['evidence_id'])
            encoded=self.tokenizer(row['text'],add_special_tokens=False,truncation=False,return_offsets_mapping=True)
            ids,offsets=encoded['input_ids'],encoded['offset_mapping']
            if len(ids)>20000: raise ValueError('Reranker source token budget exceeded')
            match=None
            if 'matched_chunk' in hit:
                if self.matched_chunk_validator is None: raise ValueError('Unverified matched chunk cannot guide reranking')
                match=self.matched_chunk_validator(hit)
            matched=match['source_char_span'] if match is not None else None
            windows,omitted=select_passage_windows(row['text'],offsets,capacity,query,matched)
            if pairs+len(windows)>max_pairs: raise ValueError('Reranker pair budget exhausted before complete candidate ranking')
            values=[]
            for window in windows:
                start,end=window['source_token_span']
                inputs=self.tokenizer.prepare_for_model(qids,pair_ids=ids[start:end],add_special_tokens=True,
                    truncation=False,return_attention_mask=True,return_token_type_ids=False)
                if len(inputs['input_ids'])>self.max_length: raise ValueError('Reranker input budget exceeded')
                tensors={k:self.torch.tensor([v],dtype=self.torch.long) for k,v in inputs.items()}
                with self.torch.inference_mode(): value=float(self.model(**tensors).logits[0,0])
                if not math.isfinite(value): raise ValueError('Nonfinite reranker logit')
                values.append(value);pairs+=1
                window['logit']=value;window['tokens_with_specials']=len(inputs['input_ids'])
            if not values: continue
            result.append({'route':'bge_reranker','score':max(values),**row,**({'matched_chunk':match} if match is not None else {})})
            notes.append({'evidence_id':row['evidence_id'],'source_windows':windows,'omitted_windows':omitted,
                          'matched_chunk':match,'matched_chunk_verified_against_index_and_source_tokens':match is not None})
        result.sort(key=lambda h:(-h['score'],h['evidence_id']))
        return result[:top_k],{'pairs_encoded':pairs,'candidate_regions':len(hits),'source_window_notes':notes,
                               'wall_ms':round((time.perf_counter()-started)*1000,3),'query_truncated':False,
                               'max_pair_budget':max_pairs,'answer_support_established':False}
