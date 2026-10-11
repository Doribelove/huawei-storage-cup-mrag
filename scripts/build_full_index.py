"""Resumable genuine CPU BGE build. This process has no question or QA input."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch
import uuid
import numpy as np
import psutil
from evidence_lab.cli import load_evidence
from evidence_lab.retrieval_policy import CandidatePolicy
from evidence_lab.safe_data import reject_answer_fields,sha256_file
from evidence_lab.semantic import LocalBGE,DenseIndex,valid_vector


def memory():
    value=psutil.Process().memory_info()
    return {'rss_bytes':value.rss,'peak_process_working_set_bytes':getattr(value,'peak_wset',None)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-dir',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--original-corpus',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-build-seconds',type=float,default=7200)
    args=parser.parse_args()
    if not 60<=args.max_build_seconds<=14400: raise ValueError('Full build wall budget must be60..14400s')
    out=args.freeze_dir;manifest_path=out/'full-manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));reject_answer_fields(manifest)
    manifest_sha=sha256_file(manifest_path);source=out/'regions.jsonl';source_sha=sha256_file(source)
    if manifest['document_count']!=100 or source_sha!=manifest['frozen_corpus_sha256']:
        raise ValueError('Frozen full source manifest mismatch')
    original_sha=sha256_file(args.original_corpus)
    if original_sha!=manifest['source_corpus_sha256']: raise ValueError('Original source corpus mismatch')
    rows=load_evidence(source);policy=CandidatePolicy(rows)
    if len(rows)!=manifest['source_regions'] or policy.report()!=manifest['policy']:
        raise ValueError('Frozen candidate policy/source mismatch')
    bydoc=defaultdict(list)
    for row in rows:
        if policy.accepts(row): bydoc[row['document_name']].append(row)
    path=out/'full.nano.json'
    if path.exists(): raise ValueError('Completed full index already exists; no overwrite')
    checkpoints=out/'document-checkpoints';checkpoints.mkdir(exist_ok=True)
    lock=out/'.full-build.lock';token=uuid.uuid4().hex
    if lock.exists():
        previous=json.loads(lock.read_text(encoding='utf-8'))
        if not args.resume or psutil.pid_exists(previous['pid']):
            raise ValueError('Full build lock exists; no duplicate start')
        lock.unlink()  # One exact task-owned stale lock, never process termination.
    with lock.open('x',encoding='utf-8') as stream:
        json.dump({'pid':psutil.Process().pid,'token':token,'manifest_sha256':manifest_sha},stream)
    try:
        with patch.object(socket.socket,'connect',side_effect=RuntimeError('Full build prohibits network')) as network:
            started=time.perf_counter();model=LocalBGE(args.model_dir)
            model_load_seconds=time.perf_counter()-started
            print(json.dumps({'stage':'full_model_ready','model_load_seconds':round(model_load_seconds,3),
                              'expected_chunks':manifest['expected_dense_chunks'],'cpu_threads':8}),flush=True)
            build_started=time.perf_counter();chunks=[];documents=[];reused=[]
            for doc in manifest['documents']:
                name=doc['name'];metadata=checkpoints/(name+'.json');vectors=checkpoints/(name+'.npy')
                if metadata.exists() or vectors.exists():
                    if not args.resume or not metadata.exists() or not vectors.exists():
                        raise ValueError('Incomplete/existing checkpoint requires inspection; no overwrite')
                    saved=json.loads(metadata.read_text(encoding='utf-8'));reject_answer_fields(saved)
                    if saved['manifest_sha256']!=manifest_sha or saved['model_identity']!=model.identity:
                        raise ValueError('Checkpoint source/model identity mismatch')
                    if sha256_file(vectors)!=saved['vectors_sha256'] or vectors.stat().st_size>128*1024**2:
                        raise ValueError('Checkpoint vector hash/byte mismatch')
                    matrix=np.load(vectors,allow_pickle=False,max_header_size=1024)
                    if matrix.dtype!=np.float32 or matrix.shape!=(len(saved['chunks']),1024):
                        raise ValueError('Checkpoint vector shape/dtype mismatch')
                    encoded=[{**item,'__vector__':valid_vector(matrix[i])} for i,item in enumerate(saved['chunks'])]
                    item=saved['statistics'];reused.append(name)
                else:
                    doc_started=time.perf_counter();encoded=[];stats=defaultdict(float)
                    for position,row in enumerate(bydoc[name],1):
                        if time.perf_counter()-build_started>args.max_build_seconds:
                            raise TimeoutError('Full build budget exhausted; completed document checkpoints preserved')
                        encoded.extend(model.region(row))
                        for key,value in model.last_region_stats.items(): stats[key]+=value
                        if position%100==0:
                            print(json.dumps({'document':name,'regions_done':position,'source_regions':len(bydoc[name]),
                                              'chunks':len(encoded),'elapsed_seconds':round(time.perf_counter()-build_started,1)}),flush=True)
                    item={'name':name,'searchable_regions':len(bydoc[name]),'chunks':len(encoded),
                          'source_tokens':int(stats['source_tokens']),'tokenization_ms':round(stats['tokenization_ms'],3),
                          'embedding_ms':round(stats['embedding_ms'],3),
                          'source_encoding_seconds':round(time.perf_counter()-doc_started,3),**memory()}
                    matrix=np.stack([c['__vector__'] for c in encoded]).astype(np.float32)
                    np.save(vectors,matrix,allow_pickle=False)
                    saved={'manifest_sha256':manifest_sha,'model_identity':model.identity,
                           'vectors_sha256':sha256_file(vectors),'statistics':item,
                           'chunks':[{k:v for k,v in c.items() if k!='__vector__'} for c in encoded]}
                    metadata.write_text(json.dumps(saved,indent=2)+'\n',encoding='utf-8')
                if item['chunks']!=doc['expected_chunks'] or item['searchable_regions']!=doc['searchable_regions'] or item['source_tokens']!=doc['source_tokens']:
                    raise ValueError('Frozen per-document encoding coverage mismatch')
                if any(c['parent_evidence_id'] not in {r['evidence_id'] for r in bydoc[name]} for c in encoded):
                    raise ValueError('Checkpoint outside source document')
                chunks.extend(encoded);documents.append(item)
                summary={'complete':False,'manifest_sha256':manifest_sha,'documents_done':len(documents),
                         'chunks_done':len(chunks),'documents':documents,'current_run_seconds':round(time.perf_counter()-build_started,3)}
                (out/'full-build-checkpoint.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
                print(json.dumps({'stage':'document_committed',**item,'documents_done':len(documents),'total_chunks':len(chunks)}),flush=True)
            encoding_wall=time.perf_counter()-build_started
            if len(chunks)!=manifest['expected_dense_chunks']: raise ValueError('Full chunk coverage mismatch')
        started=time.perf_counter();index=DenseIndex(path,rows,model.identity,chunks,source_tokenizer=model.tokenizer)
            serialization_seconds=time.perf_counter()-started
            report={'complete':True,'kind':'Genuine pinned full100-PDF CPU source embedding build; no QA inputs',
                    'manifest_sha256':manifest_sha,'frozen_corpus_sha256':source_sha,'original_corpus_sha256':original_sha,
                    'model_identity':model.identity,'document_count':len(documents),'indexed_documents':len(index.indexed_documents),
                    'source_regions':len(rows),'searchable_regions':len(policy.searchable()),'dense_chunks':index.chunk_count,
                    'model_load_seconds':round(model_load_seconds,3),'current_run_encoding_wall_seconds':round(encoding_wall,3),
                    'sum_measured_document_source_encoding_seconds':round(sum(d['source_encoding_seconds'] for d in documents),3),
                    'serialization_seconds':round(serialization_seconds,3),'resumed_documents':reused,'documents':documents,
                    'index_sha256':sha256_file(path),'index_bytes':path.stat().st_size,'old_index_vectors_reused':False,
                    'qa_questions_or_answers_read':False,'source_corpus_unchanged':sha256_file(source)==source_sha,
                    'original_corpus_unchanged':sha256_file(args.original_corpus)==original_sha,
                    'socket_connect_attempts':network.call_count,'paid_api_cost':0,**memory()}
            if network.call_count or not report['source_corpus_unchanged'] or not report['original_corpus_unchanged'] or len(index.indexed_documents)!=100:
                raise ValueError('Full build offline/source/coverage invariant failed')
            (out/'full-build-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            (out/'full-build-checkpoint.json').write_text(json.dumps({**summary,'complete':True,'index_sha256':report['index_sha256']},indent=2)+'\n',encoding='utf-8')
            print(json.dumps({k:v for k,v in report.items() if k not in ('documents','model_identity')},indent=2),flush=True)
    finally:
        if lock.exists() and json.loads(lock.read_text(encoding='utf-8')).get('token')==token:
            lock.unlink()


if __name__=='__main__': main()
