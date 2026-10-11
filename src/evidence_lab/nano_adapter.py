"""Validate JSON + base64 numeric arrays before invoking the pinned NanoVectorDB."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import zlib
import numpy as np

from .safe_data import reject_answer_fields,sha256_file


def decode_cached_vector(encoded:str,dimension:int=1024)->np.ndarray:
    if not isinstance(encoded,str) or len(encoded)>dimension*8+1024:
        raise ValueError('Compressed vector exceeds input budget')
    try: packed=base64.b64decode(encoded,validate=True)
    except Exception as error: raise ValueError('Invalid base64 vector') from error
    decompressor=zlib.decompressobj()
    raw=decompressor.decompress(packed,dimension*2+1)
    if len(raw)!=dimension*2 or not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
        raise ValueError('Wrong dimension or compressed-vector expansion exceeds budget')
    vector=np.frombuffer(raw,dtype=np.float16).astype(np.float32)
    if not np.isfinite(vector).all() or np.linalg.norm(vector)==0:
        raise ValueError('Nonfinite or zero vector')
    return vector


class ReadOnlyNanoIndex:
    def __init__(self,path:Path,dimension:int=1024,max_bytes:int=128*1024**2):
        if path.suffix.lower()!='.json' or path.stat().st_size>max_bytes:
            raise ValueError('Only bounded JSON index files accepted')
        with path.open(encoding='utf-8') as source: storage=json.load(source)
        if storage.get('embedding_dim')!=dimension: raise ValueError('Embedding dimension mismatch')
        records=storage.get('data')
        if not isinstance(records,list) or not 0<len(records)<=100000:
            raise ValueError('Invalid/budget-exceeding record list')
        seen=set()
        for record in records:
            if not isinstance(record,dict) or not isinstance(record.get('__id__'),str): raise ValueError('Invalid index ID')
            if record['__id__'] in seen: raise ValueError('Duplicate index ID')
            seen.add(record['__id__'])
            reject_answer_fields(record)
        encoded=storage.get('matrix')
        expected=len(records)*dimension*4
        if not isinstance(encoded,str) or len(encoded)>4*((expected+2)//3):
            raise ValueError('Missing/oversized matrix encoding')
        try: matrix_bytes=base64.b64decode(encoded,validate=True)
        except Exception as error: raise ValueError('Invalid base64 matrix') from error
        if len(matrix_bytes)!=expected: raise ValueError('Matrix dimension/count mismatch')
        matrix=np.frombuffer(matrix_bytes,dtype=np.float32).reshape(len(records),dimension)
        if not np.isfinite(matrix).all() or (np.linalg.norm(matrix,axis=1)==0).any():
            raise ValueError('Nonfinite or zero matrix rows')
        self.path,self.dimension=path,dimension
        self.records=records
        self.matrix=matrix
        self.before_hash=sha256_file(path)
        # A trusted PyPI dependency, pinned to the provided uv.lock version, not imported archive code.
        from nano_vectordb import NanoVectorDB
        self.database=NanoVectorDB(dimension,storage_file=str(path))

    def query_vector(self,query:np.ndarray,top_k:int=5,threshold:float|None=None)->list[dict]:
        query=np.asarray(query,dtype=np.float32)
        if query.shape!=(self.dimension,) or not np.isfinite(query).all() or np.linalg.norm(query)==0:
            raise ValueError('Invalid query vector shape/value')
        if not 1<=top_k<=100: raise ValueError('top_k must be 1..100')
        results=self.database.query(query,top_k=top_k,better_than_threshold=threshold)
        return [{'id':r['__id__'],'similarity':float(r['__metrics__']),'file_path':r.get('file_path'),
                 'content':r.get('content',''),'full_doc_id':r.get('full_doc_id')} for r in results]

    def replay_record(self,index:int=0,top_k:int=5)->dict:
        if not 0<=index<len(self.records): raise ValueError('Record index out of bounds')
        record=self.records[index]
        hits=self.query_vector(self.matrix[index],top_k)
        return {'query_source':'existing_document_embedding; not a question embedding',
                'query_id':record['__id__'],'self_in_top_k':any(h['id']==record['__id__'] for h in hits),
                'results':[{k:v for k,v in h.items() if k!='content'} for h in hits]}

    def assert_unchanged(self)->None:
        if sha256_file(self.path)!=self.before_hash: raise RuntimeError('Source index changed during read-only replay')
