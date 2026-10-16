"""Read-only import preflight using the same interpreter and -I worker isolation."""
import argparse
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
import sysconfig

project=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--role',choices=('client','worker'),required=True)
args=ap.parse_args()
mode={'legacy_vendor_enabled':False}
if args.role=='client':
    spec=importlib.util.spec_from_file_location('release_dependencies',project/'rag-anything-experiments/dependency_loader.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    mode=module.configure_dependencies(project)
modules = ('dotenv','requests','openai','pydantic','pydantic_core','jiter') if args.role=='client' else ('numpy','PIL','pdfplumber','pypdf','pypdfium2','nano_vectordb','psutil','torch','torchvision','transformers','safetensors')
result={'role':args.role,'python':sys.version.split()[0],'executable':sys.executable,
        'platform':platform.system()+'/'+platform.machine(),'SOABI':sysconfig.get_config_var('SOABI'),
        'isolated':bool(sys.flags.isolated),'dependency_policy':mode,'modules':{},'models_loaded':False}
for name in modules:
    try:
        value=importlib.import_module(name)
        result['modules'][name]={'ok':True,'file':getattr(value,'__file__',None),'version':getattr(value,'__version__',None)}
    except Exception as exc:result['modules'][name]={'ok':False,'error':type(exc).__name__+': '+str(exc)}
result['passed']=all(v['ok'] for v in result['modules'].values())
print(json.dumps(result,ensure_ascii=False,indent=2))
raise SystemExit(0 if result['passed'] else 1)
