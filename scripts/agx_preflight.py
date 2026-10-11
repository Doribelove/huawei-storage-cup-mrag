#!/usr/bin/env python3
"""Local read-only Jetson inventory. Python3.8+, stdlib only; never a model-readiness test."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

PACKAGES = ('torch','torchvision','transformers','tokenizers','safetensors','numpy',
    'Pillow','accelerate','vllm','flash-attn','qwen-vl-utils','nano-vectordb','tensorrt','onnxruntime')
DEBIAN_PACKAGES = ('nvidia-jetpack','nvidia-l4t-core','nvidia-l4t-cuda',
    'nvidia-container-runtime','nvidia-container-toolkit','nvidia-tensorrt')
READ_FILES = ('/proc/device-tree/model','/sys/firmware/devicetree/base/model',
    '/etc/os-release','/etc/nv_tegra_release','/proc/meminfo',
    '/usr/local/cuda/version.json','/usr/local/cuda/version.txt')
MAX_BYTES = 16384

# Only the selected existing interpreter is inspected; no transformers/model import.
PYTHON_REPORT_CODE = r'''
import contextlib,importlib.metadata as m,io,json,os,socket,sys
blocked_actions=[]
def guard(event,args):
    if event.startswith('socket.') or event in ('subprocess.Popen','os.system','os.mkdir','os.remove','os.rename','os.rmdir'):
        category='python_socket_operation' if event.startswith('socket.') else 'python_process_or_filesystem_mutation'
        if category not in blocked_actions: blocked_actions.append(category)
        raise RuntimeError('Read-only Python probe blocked an action')
    if event=='open':
        mode=args[1]; flags=args[2]
        if (isinstance(mode,str) and any(x in mode for x in 'wax+')) or (
            isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
            if 'python_file_write' not in blocked_actions: blocked_actions.append('python_file_write')
            raise RuntimeError('Read-only Python probe blocked a write')
sys.addaudithook(guard)
packages=json.loads(sys.argv[2]); versions={}
for name in packages:
    try: versions[name]={'status':'installed_metadata','version':m.version(name)}
    except m.PackageNotFoundError: versions[name]={'status':'missing'}
    except Exception: versions[name]={'status':'metadata_error'}
report={'python_version':'.'.join(str(x) for x in sys.version_info[:3]),'packages':versions,
    'torch_runtime':{'status':'not_requested'},'model_loaded':False,'model_generate_called':False}
if sys.argv[1]=='1':
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            import torch
            available=bool(torch.cuda.is_available())
            devices=[]
            if available:
                for index in range(min(int(torch.cuda.device_count()),8)):
                    props=torch.cuda.get_device_properties(index)
                    devices.append({'name':props.name,'compute_capability':[int(props.major),int(props.minor)],
                        'total_memory_bytes':int(props.total_memory)})
        report['torch_runtime']={'status':'queried','version':str(torch.__version__),
            'cuda_build_version':torch.version.cuda,'cuda_available':available,'devices':devices,
            'cuda_kernel_execution_tested':False,'bf16_execution_tested':False}
    except Exception as exc:
        report['torch_runtime']={'status':'import_or_query_failed','error_type':type(exc).__name__}
report['torch_runtime']['blocked_actions']=blocked_actions
print(json.dumps(report))
'''


def version(value):
    if isinstance(value,str) and re.fullmatch(r'[0-9][A-Za-z0-9.+~:_-]{0,79}',value): return value
    return None


def jetson_model(value):
    # Output only known model tokens, never raw firmware text or appended identifiers.
    found = re.search(r'\bJetson\s+(AGX\s+)?(Orin|Xavier)(\s+(?:NX|Nano))?\b',value,re.I)
    if not found: return None
    return 'NVIDIA Jetson '+('AGX ' if found[1] else '')+found[2].title()+(
        ' '+found[3].strip().upper().replace('NANO','Nano') if found[3] else '')


def safe_environment():
    # Do not copy credential/token variables into the child or record environment values.
    result={key:os.environ[key] for key in ('PATH','LD_LIBRARY_PATH','SYSTEMROOT','WINDIR') if key in os.environ}
    result.update(LC_ALL='C',LANG='C',PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    return result


class LocalHost:
    def read(self,path):
        if path not in READ_FILES: raise ValueError('File is outside the inventory allowlist')
        try:
            with open(path,'rb') as stream: raw=stream.read(MAX_BYTES+1)
            if len(raw)>MAX_BYTES: return {'status':'too_large'}
            return {'status':'read','text':raw.decode('utf-8',errors='replace').replace('\0','\n')}
        except FileNotFoundError: return {'status':'missing'}
        except PermissionError: return {'status':'permission_denied'}
        except OSError: return {'status':'read_error'}

    def run(self,name,args,timeout):
        if name not in ('nvcc','dpkg-query','nvidia-smi','python_inventory'):
            raise ValueError('Unknown inventory command')
        try:
            result=subprocess.run(args,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,timeout=timeout,shell=False,env=safe_environment())
        except FileNotFoundError: return {'status':'missing'}
        except PermissionError: return {'status':'permission_denied'}
        except subprocess.TimeoutExpired: return {'status':'timeout'}
        except OSError: return {'status':'launch_error'}
        if len(result.stdout)>MAX_BYTES: return {'status':'output_too_large','exit_code':result.returncode}
        # Raw stderr is deliberately omitted, including paths/serials/credentials in errors.
        return {'status':'completed' if result.returncode==0 else 'command_failed',
            'exit_code':result.returncode,'text':result.stdout.decode('utf-8',errors='replace')}

    def disk(self,path):
        try:
            stats=os.statvfs(path)
            return {'status':'read','total_bytes':stats.f_frsize*stats.f_blocks,
                'available_bytes':stats.f_frsize*stats.f_bavail}
        except (OSError,AttributeError): return {'status':'unavailable'}


def parse_os(text):
    values={}
    for line in text.splitlines():
        key,sep,value=line.partition('='); value=value.strip().strip('"\'')
        if sep and key=='ID' and re.fullmatch(r'[a-z][a-z0-9_-]{0,31}',value): values['id']=value
        if sep and key=='VERSION_ID' and re.fullmatch(r'[0-9]{1,4}(?:\.[0-9]{1,4}){0,2}',value):
            values['version_id']=value
    return values


def parse_memory(text):
    return {name.lower()+'_bytes':int(count)*1024 for name,count in re.findall(
        r'^(MemTotal|MemAvailable|SwapTotal|SwapFree):\s+(\d+)\s+kB\s*$',text,re.M)}


def parse_python(text):
    try: raw=json.loads(text)
    except (ValueError,TypeError): return {'status':'invalid_report'}
    if not isinstance(raw,dict) or not version(raw.get('python_version')):
        return {'status':'invalid_report'}
    packages={}
    for name in PACKAGES:
        item=raw.get('packages',{}).get(name,{}) if isinstance(raw.get('packages'),dict) else {}
        parsed=version(item.get('version')) if isinstance(item,dict) else None
        packages[name]={'status':'installed_metadata','version':parsed} if parsed else {
            'status':item.get('status') if isinstance(item,dict) and item.get('status') in ('missing','metadata_error') else 'invalid_metadata'}
    runtime=raw.get('torch_runtime',{}); safe={'status':'invalid_report'}
    if isinstance(runtime,dict) and runtime.get('status') in ('not_requested','import_or_query_failed'):
        safe={'status':runtime['status']}  # No arbitrary error message/type copied.
    elif isinstance(runtime,dict) and runtime.get('status')=='queried' and type(runtime.get('cuda_available')) is bool:
        safe={'status':'queried','version':version(runtime.get('version')),
            'cuda_build_version':version(runtime.get('cuda_build_version')),'cuda_available':runtime['cuda_available'],
            'devices':[],'cuda_kernel_execution_tested':False,'bf16_execution_tested':False}
        devices=runtime.get('devices',[])
        for device in devices[:8] if isinstance(devices,list) else []:
            if not isinstance(device,dict): continue
            capability=device.get('compute_capability'); total=device.get('total_memory_bytes')
            if not (isinstance(capability,list) and len(capability)==2 and all(type(x) is int and 0<=x<=99 for x in capability)
                    and type(total) is int and 0<=total<2**60): continue
            name=str(device.get('name','')); identified=jetson_model('Jetson '+name)
            safe['devices'].append({'model':identified or 'GPU name omitted: not a recognized Jetson model',
                'compute_capability':capability,'total_memory_bytes':total})
    if isinstance(runtime,dict):
        blocked=runtime.get('blocked_actions',[])
        safe['blocked_actions']=sorted({x for x in blocked if isinstance(x,str) and x in (
            'python_socket_operation','python_file_write','python_process_or_filesystem_mutation')}) if isinstance(blocked,list) else []
    return {'status':'collected','python_version':raw['python_version'],'packages':packages,
        'metadata_proves_import_or_model_compatibility':False,'torch_runtime':safe,
        'model_loaded':False,'model_generate_called':False}


def collect(host,*,python=None,probe_torch=False,timeout=15,disk_path='/',system=None,machine=None):
    system=system or platform.system(); machine=machine or platform.machine()
    linux=system=='Linux'; warnings=[]; reads={}; commands={}
    def read(path):
        result=host.read(path) if linux else {'status':'not_linux'}
        reads[path]=result['status']; return result.get('text','')
    def run(name,args):
        result=host.run(name,args,timeout); commands[name]={k:result[k] for k in ('status','exit_code') if k in result}
        return result
    model=jetson_model(read(READ_FILES[0])) or jetson_model(read(READ_FILES[1]))
    osinfo=parse_os(read('/etc/os-release'))
    l4ttext=read('/etc/nv_tegra_release')
    l4t=re.search(r'\bR(\d+)\b.*?REVISION:\s*(\d+(?:\.\d+)*)',l4ttext,re.S)
    l4tinfo={'release':l4t[1],'revision':l4t[2]} if l4t else None
    memory=parse_memory(read('/proc/meminfo'))
    try: cuda=version(json.loads(read('/usr/local/cuda/version.json')).get('cuda',{}).get('version'))
    except (ValueError,TypeError,AttributeError): cuda=None
    cudatext=read('/usr/local/cuda/version.txt')
    if not cuda:
        found=re.search(r'CUDA Version\s+(\d+(?:\.\d+){1,2})',cudatext,re.I)
        cuda=found[1] if found else None
    debian={name:{'status':'unavailable'} for name in DEBIAN_PACKAGES}; gpu=[]
    if linux:
        result=run('nvcc',['nvcc','--version'])
        found=re.search(r'\brelease\s+(\d+\.\d+)',result.get('text',''))
        nvcc=found[1] if found and result['status']=='completed' else None
        result=run('dpkg-query',['dpkg-query','-W','-f=${binary:Package}\t${Version}\t${db:Status-Abbrev}\n',*DEBIAN_PACKAGES])
        for line in result.get('text','').splitlines():
            parts=line.split('\t')
            if len(parts)!=3: continue
            name=parts[0].split(':')[0]; installed=parts[2].startswith('ii'); parsed=version(parts[1])
            if name in debian: debian[name]={'status':'installed' if installed and parsed else 'not_installed',
                'version':parsed if installed else None}
        result=run('nvidia-smi',['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free',
            '--format=csv,noheader,nounits'])
        if result['status']=='completed':
            for line in result.get('text','').splitlines()[:8]:
                fields=[x.strip() for x in line.split(',')]
                if len(fields)==4 and version(fields[1]) and all(re.fullmatch(r'\d+',x) for x in fields[2:]):
                    gpu.append({'model':jetson_model('Jetson '+fields[0]) or 'Non-Jetson GPU name omitted',
                        'driver_version':fields[1],'memory_total_bytes':int(fields[2])*1024**2,
                        'memory_free_bytes':int(fields[3])*1024**2})
    else: nvcc=None
    disk=host.disk(disk_path)
    if python:
        result=run('python_inventory',[python,'-I','-B','-c',PYTHON_REPORT_CODE,
            '1' if probe_torch else '0',json.dumps(PACKAGES)])
        pythoninfo=parse_python(result.get('text','')) if result['status']=='completed' else {'status':result['status']}
    else: pythoninfo={'status':'missing_interpreter'}
    if not linux: warnings.append('not_linux_no_jetson_host_verified')
    if machine not in ('aarch64','arm64'): warnings.append('aarch64_not_observed')
    for key,present in (('model_unknown',model),('os_version_unknown',osinfo.get('version_id')),
            ('l4t_unknown',l4tinfo),('memory_unknown',memory.get('memtotal_bytes')),
            ('disk_unknown',disk.get('status')=='read'),('cuda_toolkit_version_unknown',cuda or nvcc),
            ('python_inventory_incomplete',pythoninfo.get('status')=='collected')):
        if not present: warnings.append(key)
    for name,result in commands.items():
        if result['status']!='completed': warnings.append(name+'_'+result['status'])
    for name,state in reads.items():
        if state in ('permission_denied','read_error','too_large'): warnings.append(name+'_'+state)
    torch_complete=not probe_torch or pythoninfo.get('torch_runtime',{}).get('status')=='queried'
    if not torch_complete: warnings.append('requested_torch_runtime_probe_incomplete')
    complete=linux and machine in ('aarch64','arm64') and model and l4tinfo and osinfo.get('version_id') and (
        memory.get('memtotal_bytes') and disk.get('status')=='read' and (cuda or nvcc) and pythoninfo.get('status')=='collected' and torch_complete)
    code=0 if complete else 2
    return {'schema_version':1,'collected_utc':datetime.now(timezone.utc).isoformat(),
        'status':'inventory_collected_compatibility_unverified' if complete else 'partial_inventory',
        'inventory_exit_code':code,'exit_code_contract':'0=inventory collected;2=missing inventory/not Jetson Linux; neither proves model readiness',
        'platform':{'system':'Linux' if linux else 'not_linux','architecture':machine if machine in (
            'aarch64','arm64','x86_64','AMD64','armv7l') else 'unknown_redacted','model':model,'os':osinfo,'l4t':l4tinfo},
        'debian_packages':debian,'cuda_toolkit':{'version_file':cuda,'nvcc_release':nvcc,
            'toolkit_version_is_not_torch_cuda_build':True},'gpu_query':gpu,'memory':memory,'disk':disk,
        'disk_path_recorded':False,'python':pythoninfo,'source_status':reads,'command_status':commands,
        'warnings':sorted(set(warnings)),'probe_torch_requested':probe_torch,
        'compatibility':{'application_verified':False,'weight_memory_fit_verified':False,
            'visual_processor_on_target_verified':False,'model_generation_verified':False,
            'shared_jetson_memory_must_not_be_added_as_separate_host_and_gpu_capacity':True},
        'scope':{'local_only':True,'network_scan':False,'download_or_install':False,'model_load_or_generate':False,
            'script_writes_files':False,'raw_command_stderr_recorded':False,
            'serial_mac_hostnames_usernames_credentials_collected':False}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',help='Existing local interpreter to inspect; default is this interpreter')
    parser.add_argument('--probe-torch',action='store_true',help='Opt-in existing torch import and CUDA property queries; no tensor/model/kernel test')
    parser.add_argument('--disk-path',default='/',help='Existing local target directory, read statvfs only; path omitted from report')
    parser.add_argument('--timeout',type=int,default=15,help='Per-command timeout1..30 seconds')
    args=parser.parse_args(argv)
    if not 1<=args.timeout<=30: parser.error('--timeout must be1..30')
    if args.disk_path.startswith(('//','\\\\')) or re.match(r'^[A-Za-z]+://',args.disk_path):
        parser.error('--disk-path must be local; network/URI paths are forbidden')
    if not Path(args.disk_path).is_dir(): parser.error('--disk-path must be an existing local directory')
    candidate=args.python or sys.executable
    if candidate.startswith(('//','\\\\')) or re.match(r'^[A-Za-z]+://',candidate):
        parser.error('--python must name an existing local interpreter; network/URI paths are forbidden')
    python=shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
    result=collect(LocalHost(),python=python,probe_torch=args.probe_torch,timeout=args.timeout,disk_path=args.disk_path)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return result['inventory_exit_code']


if __name__=='__main__': raise SystemExit(main())
