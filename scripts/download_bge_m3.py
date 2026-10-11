"""Download only six pinned public BAAI files; no remote code or Hub login."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.request import urlopen

REVISION='5617a9f61b028005a4858fdac845db406aefb181'
FILES={
 'config.json':(687,'git_blob_sha1','e6eda1c72da8f9dc30fdd9b69c73d35af3b7a7ad'),
 'tokenizer_config.json':(444,'git_blob_sha1','dc69ac559dcba2694012009aaa108c614541789a'),
 'special_tokens_map.json':(964,'git_blob_sha1','b1879d702821e753ffe4245048eee415d54a9385'),
 'sentencepiece.bpe.model':(5069051,'sha256','cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865'),
 'tokenizer.json':(17098108,'sha256','21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08'),
 'pytorch_model.bin':(2271145830,'sha256','b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38')}


def hashes(path:Path)->dict:
    size=path.stat().st_size
    sha256=hashlib.sha256()
    git_sha1=hashlib.sha1(b'blob '+str(size).encode()+b'\0')
    with path.open('rb') as source:
        while chunk:=source.read(8*1024**2): sha256.update(chunk);git_sha1.update(chunk)
    return {'bytes':size,'sha256':sha256.hexdigest(),'git_blob_sha1':git_sha1.hexdigest()}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('models/bge-m3'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.output).free<8*1024**3: raise ValueError('At least 8GiB free space required before model preparation')
    started=time.perf_counter()
    manifest=[]
    for name,(size,kind,expected) in FILES.items():
        target=args.output/name
        if target.exists():
            actual=hashes(target)
            if actual['bytes']!=size or actual[kind]!=expected: raise ValueError(f'Existing {name} differs; refusing overwrite')
            manifest.append({'name':name,**actual,'expected_hash_kind':kind,'expected_hash':expected})
            continue
        partial=args.output/(name+'.partial')
        url=f'https://huggingface.co/BAAI/bge-m3/resolve/{REVISION}/{name}'
        done=0
        # TLS and redirects use the standard verifier; never disable certificate checks.
        with urlopen(url,timeout=60) as response,partial.open('wb') as destination:
            next_progress=64*1024**2
            while chunk:=response.read(4*1024**2):
                done+=len(chunk)
                if done>size: raise ValueError(f'{name} exceeds fixed download size')
                destination.write(chunk)
                if done>=next_progress:
                    print(f'{name}: {done}/{size} bytes ({100*done/size:.1f}%)',flush=True)
                    next_progress+=64*1024**2
        actual=hashes(partial)
        if actual['bytes']!=size or actual[kind]!=expected: raise ValueError(f'{name} failed official size/hash verification')
        partial.rename(target)
        manifest.append({'name':name,**actual,'expected_hash_kind':kind,'expected_hash':expected})
        print(f'{name}: size/hash verified',flush=True)
    report={'model_id':'BAAI/bge-m3','revision':REVISION,'files':manifest,
            'downloaded_bytes':sum(r['bytes'] for r in manifest),'seconds':round(time.perf_counter()-started,3),
            'remote_code_downloaded':False,'trust_remote_code':False,'no_login_or_paid_api':True,
            'safetensors_source_note':'Pinned official main contains no safetensors; convert hash-verified bin locally with torch weights_only=True before model loading.'}
    (args.output/'source-manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))


if __name__=='__main__': main()
