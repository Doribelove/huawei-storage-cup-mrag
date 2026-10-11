from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from .lexical import BM25
from .material_audit import audit_archives
from .nano_adapter import ReadOnlyNanoIndex,decode_cached_vector
from .safe_data import reject_answer_fields,sha256_file


def load_evidence(path:Path)->list[dict]:
    if path.stat().st_size>128*1024**2: raise ValueError('Corpus exceeds budget')
    records=[]
    with path.open(encoding='utf-8') as source:
        for line in source:
            record=json.loads(line)
            reject_answer_fields(record)
            if record.get('role')!='source_pdf_region':
                raise ValueError('Only source PDF region evidence can be searched')
            records.append(record)
            if len(records)>50000: raise ValueError('Evidence count budget exceeded')
    return records


def main():
    parser=argparse.ArgumentParser(description='Network-free material audit and genuine index replay')
    commands=parser.add_subparsers(dest='command',required=True)
    audit=commands.add_parser('audit')
    audit.add_argument('--data-zip',type=Path,required=True)
    audit.add_argument('--demo-zip',type=Path,required=True)
    audit.add_argument('--output',type=Path,required=True)
    probe=commands.add_parser('replay')
    probe.add_argument('--index',type=Path,required=True)
    probe.add_argument('--output',type=Path,required=True)
    probe.add_argument('--probes',type=int,default=10)
    search=commands.add_parser('search')
    search.add_argument('--corpus',type=Path,required=True)
    search.add_argument('--query',required=True)
    search.add_argument('--top-k',type=int,default=5)
    args=parser.parse_args()
    if args.command=='audit':
        report=audit_archives(args.data_zip,args.demo_zip,args.output)
        print(json.dumps({'dataset':{k:v for k,v in report['dataset'].items() if k!='documents'},
                          'status_reconciliation':report['status_reconciliation'],'runtime_seconds':report['runtime_seconds']},ensure_ascii=False,indent=2))
    elif args.command=='replay':
        if not 1<=args.probes<=100: raise ValueError('Probes must be 1..100')
        begin=time.perf_counter()
        index=ReadOnlyNanoIndex(args.index)
        load_seconds=time.perf_counter()-begin
        vector_checks=0
        for record in index.records:
            if 'vector' in record:
                decode_cached_vector(record['vector'],index.dimension)
                vector_checks+=1
        results=[]
        latencies=[]
        for i in range(min(args.probes,len(index.records))):
            position=i*(len(index.records)-1)//max(args.probes-1,1)
            begin=time.perf_counter()
            results.append(index.replay_record(position))
            latencies.append((time.perf_counter()-begin)*1000)
        index.assert_unchanged()
        names={r.get('file_path') for r in index.records}
        report={'test_kind':'existing-document-vector replay; not natural-question retrieval or QA evaluation',
             'index_sha256':index.before_hash,'dimension':index.dimension,'record_count':len(index.records),
             'record_keys':sorted({k for r in index.records for k in r}),'unique_file_paths':len(names),
             'compressed_record_vectors_validated':vector_checks,'load_seconds':round(load_seconds,4),
             'probes':results,'self_hits_in_top5':sum(r['self_in_top_k'] for r in results),
             'mean_probe_ms':round(sum(latencies)/len(latencies),4),'max_probe_ms':round(max(latencies),4),
             'source_index_unchanged':True,'model_calls':0,'network_calls':0,'new_question_embedding_not_run':True,
             'runtime':{'python':platform.python_version(),'nano_vectordb':importlib.metadata.version('nano-vectordb')},
             'limits':['No answer accuracy or genuine query Recall is measured','Stored model identity is not established by dimension alone']}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('probes',)},ensure_ascii=False,indent=2))
    else:
        begin=time.perf_counter()
        records=load_evidence(args.corpus)
        hits=BM25(records).search(args.query,args.top_k)
        print(json.dumps({'kind':'lexical evidence retrieval; no generated answer','query':args.query,
            'latency_ms':round((time.perf_counter()-begin)*1000,3),'corpus_sha256':sha256_file(args.corpus),
            'model_calls':0,'hits':hits},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
