"""Convert the pinned official tensor checkpoint locally; never load arbitrary pickle objects."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import psutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from download_bge_m3 import FILES,REVISION,hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,default=Path('models/bge-m3'))
    args=parser.parse_args()
    started=time.perf_counter()
    for name,(size,kind,expected) in FILES.items():
        actual=hashes(args.model_dir/name)
        if actual['bytes']!=size or actual[kind]!=expected: raise ValueError(f'Pinned source mismatch: {name}')
    target=args.model_dir/'model.safetensors'
    if target.exists(): raise ValueError('Existing converted weights: refusing overwrite; inspect conversion-manifest.json')
    state=torch.load(args.model_dir/'pytorch_model.bin',map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not state or any(not isinstance(k,str) or not isinstance(v,torch.Tensor) for k,v in state.items()):
        raise ValueError('Expected a string-to-tensor-only state dictionary')
    if any(v.device.type!='cpu' or not v.is_contiguous() for v in state.values()): raise ValueError('Unexpected tensor device/layout')
    partial=args.model_dir/'model.safetensors.partial'
    save_file(state,str(partial),metadata={'format':'pt','model_id':'BAAI/bge-m3','revision':REVISION})
    # mmap readback compares every tensor, including dtype and shape; no quantization.
    with safe_open(partial,framework='pt',device='cpu') as converted:
        if set(converted.keys())!=set(state): raise ValueError('Converted tensor key mismatch')
        for key,value in state.items():
            other=converted.get_tensor(key)
            if other.dtype!=value.dtype or other.shape!=value.shape or not torch.equal(other,value):
                raise ValueError(f'Converted tensor changed: {key}')
    partial.rename(target)
    memory=psutil.Process().memory_info()
    report={'model_id':'BAAI/bge-m3','revision':REVISION,'source_weights_sha256':FILES['pytorch_model.bin'][2],
        'converted_weights':hashes(target),'conversion':'torch.load(weights_only=True, map_location=cpu) then safetensors.save_file',
        'tensor_count':len(state),'tensor_elements':sum(v.numel() for v in state.values()),
        'all_tensor_values_shapes_dtypes_equal':True,'remote_code':False,'quantized':False,
        'seconds':round(time.perf_counter()-started,3),'rss_bytes':memory.rss,'peak_working_set_bytes':getattr(memory,'peak_wset',None),
        'torch_version':torch.__version__}
    (args.model_dir/'conversion-manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
