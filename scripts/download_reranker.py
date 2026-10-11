"""Pinned public BAAI safetensors; no login, credentials, remote code or paid API."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import time
from urllib.request import Request,urlopen
from download_bge_m3 import hashes

REVISION='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
FILES={
    'config.json':(795,'git_blob_sha1','9f62673cb00ec41dcec8947b9ed16f6f2eb23ba2'),
    'model.safetensors':(2271071852,'sha256','d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286'),
    'special_tokens_map.json':(964,'git_blob_sha1','b1879d702821e753ffe4245048eee415d54a9385'),
    'tokenizer.json':(17098273,'sha256','69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15'),
    'tokenizer_config.json':(1173,'git_blob_sha1','328a00a9a560aadcf2a3064f917517359eb3cc26')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('models/bge-reranker-v2-m3'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.output).free<6*1024**3: raise ValueError('Need6GiB free before model download')
    started=time.perf_counter();manifest=[]
    for name,(size,kind,expected) in FILES.items():
        target=args.output/name
        if target.exists():
            actual=hashes(target)
            if actual['bytes']!=size or actual[kind]!=expected: raise ValueError('Existing pinned model file mismatch; no overwrite')
        else:
            partial=args.output/(name+'.partial');done=partial.stat().st_size if partial.exists() else 0
            if done and not args.resume: raise ValueError('Partial download exists; explicit --resume required')
            if done>size: raise ValueError('Partial exceeds expected bytes')
            if done<size:
                url=f'https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/{REVISION}/{name}'
                headers={'Range':f'bytes={done}-'} if done else {}
                with urlopen(Request(url,headers=headers),timeout=30) as response:
                    if done and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {done}-')):
                        raise ValueError('Server did not honor exact resume range; no overwrite')
                    with partial.open('ab' if done else 'xb') as destination:
                        next_progress=done+64*1024**2
                        while block:=response.read(4*1024**2):
                            done+=len(block)
                            if done>size: raise ValueError('Download exceeds pinned file size')
                            destination.write(block)
                            if done>=next_progress:
                                print(f'{name}: {done}/{size}bytes',flush=True);next_progress+=64*1024**2
            actual=hashes(partial)
            if actual['bytes']!=size or actual[kind]!=expected: raise ValueError('Official size/hash verification failed')
            partial.rename(target)
        manifest.append({'name':name,**actual,'expected_hash_kind':kind,'expected_hash':expected})
        print(f'{name}: pinned size/hash verified',flush=True)
    report={'model_id':'BAAI/bge-reranker-v2-m3','revision':REVISION,'files':manifest,
            'bytes':sum(f['bytes'] for f in manifest),'preparation_seconds':round(time.perf_counter()-started,3),
            'trust_remote_code':False,'credential_or_paid_api_use':False}
    (args.output/'source-manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2),flush=True)


if __name__=='__main__': main()
