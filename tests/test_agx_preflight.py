"""Offline fixtures and this Windows host only; never an AGX compatibility claim."""
import ast
import contextlib
import importlib.util
import json
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/agx_preflight.py'
spec=importlib.util.spec_from_file_location('agx_preflight',SCRIPT)
probe=importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)


def python_fixture(runtime=None):
    return {'python_version':'3.8.10','packages':{name:{'status':'missing'} for name in probe.PACKAGES},
        'torch_runtime':runtime or {'status':'not_requested'},'model_loaded':False,'model_generate_called':False}


class HostFixture:
    def __init__(self):
        self.reads=[]; self.commands=[]
        self.files={'/proc/device-tree/model':'NVIDIA Jetson AGX Orin Developer Kit\0serial SECRET123\0',
            '/etc/os-release':'ID=ubuntu\nVERSION_ID="20.04"\nPRETTY_NAME="private-owner-PC"\n',
            '/etc/nv_tegra_release':'# R35 (release), REVISION: 6.0, BOARD: t234ref, EABI: aarch64',
            '/proc/meminfo':'MemTotal: 65536000 kB\nMemAvailable: 40000000 kB\nSwapTotal: 1000 kB\nSwapFree: 500 kB\n',
            '/usr/local/cuda/version.json':'{"cuda":{"version":"11.4.19"}}'}
        self.results={'nvcc':{'status':'missing'},'dpkg-query':{'status':'command_failed','exit_code':1,
            'text':'nvidia-jetpack\t5.1.4-b17\tii \n'},'nvidia-smi':{'status':'missing'},
            'python_inventory':{'status':'completed','exit_code':0,'text':json.dumps(python_fixture())}}
    def read(self,path):
        self.reads.append(path)
        return {'status':'read','text':self.files[path]} if path in self.files else {'status':'missing'}
    def run(self,name,args,timeout): self.commands.append((name,args,timeout)); return self.results[name]
    def disk(self,path): return {'status':'read','total_bytes':10**12,'available_bytes':10**11}


def inventory(host=None,**kwargs):
    return probe.collect(host or HostFixture(),python=sys.executable,system='Linux',machine='aarch64',**kwargs)


class AgxPreflightTests(unittest.TestCase):
    def test_known_jetson_fixture_collects_inventory_without_claiming_model_readiness(self):
        report=inventory()
        self.assertEqual(report['inventory_exit_code'],0)
        self.assertEqual(report['platform']['model'],'NVIDIA Jetson AGX Orin')
        self.assertEqual(report['platform']['os'],{'id':'ubuntu','version_id':'20.04'})
        self.assertEqual(report['platform']['l4t'],{'release':'35','revision':'6.0'})
        self.assertEqual(report['memory']['memavailable_bytes'],40000000*1024)
        self.assertEqual(report['debian_packages']['nvidia-jetpack']['version'],'5.1.4-b17')
        self.assertFalse(report['compatibility']['application_verified'])
        self.assertFalse(report['compatibility']['weight_memory_fit_verified'])
        self.assertEqual(report['python']['torch_runtime']['status'],'not_requested')

    def test_xavier_and_missing_data_do_not_infer_orin_or_64gb(self):
        host=HostFixture(); host.files['/proc/device-tree/model']='NVIDIA Jetson AGX Xavier\0'
        del host.files['/proc/meminfo']
        report=inventory(host)
        self.assertEqual(report['platform']['model'],'NVIDIA Jetson AGX Xavier')
        self.assertEqual(report['inventory_exit_code'],2)
        self.assertIn('memory_unknown',report['warnings'])
        self.assertEqual(report['memory'],{})

    def test_absent_commands_and_timeout_are_clear_without_raw_stderr(self):
        host=HostFixture(); host.files.clear()
        host.results['nvcc']={'status':'timeout'}
        host.results['python_inventory']={'status':'launch_error'}
        report=inventory(host)
        self.assertEqual(report['inventory_exit_code'],2)
        self.assertIn('nvcc_timeout',report['warnings'])
        self.assertEqual(report['python']['status'],'launch_error')
        self.assertNotIn('text',report['command_status']['dpkg-query'])

    def test_no_identifiers_environment_values_or_arbitrary_python_fields_leak(self):
        host=HostFixture(); data=python_fixture({'status':'queried','version':'2.1.0a0+nv23.06',
            'cuda_build_version':'11.4','cuda_available':True,'devices':[{'name':'Orin UUID SECRET123',
                'compute_capability':[8,7],'total_memory_bytes':64*1024**3}],
            'serial':'SECRET123','credential':'SECRET123'})
        data['packages']['torch']={'status':'installed_metadata','version':'SECRET123'}
        data['private_environment']='SECRET123'
        host.results['python_inventory']['text']=json.dumps(data)
        report=inventory(host,probe_torch=True); encoded=json.dumps(report)
        self.assertNotIn('SECRET123',encoded); self.assertNotIn('private-owner-PC',encoded)
        self.assertEqual(report['python']['torch_runtime']['devices'][0]['model'],'NVIDIA Jetson Orin')
        self.assertEqual(report['python']['packages']['torch']['status'],'invalid_metadata')

    def test_local_file_reads_and_command_names_stay_on_allowlist(self):
        host=HostFixture(); inventory(host)
        self.assertTrue(set(host.reads)<=set(probe.READ_FILES))
        self.assertEqual([name for name,_,_ in host.commands],['nvcc','dpkg-query','nvidia-smi','python_inventory'])
        args=host.commands[-1][1]
        self.assertEqual(args[:3],[sys.executable,'-I','-B'])
        self.assertEqual(args[-2],'0')
        for path in ('/proc/device-tree/serial-number','/sys/class/net/eth0/address','/home/user/.ssh/id_rsa'):
            with self.assertRaises(ValueError): probe.LocalHost().read(path)

    def test_non_linux_skips_all_linux_file_and_external_hardware_commands(self):
        host=HostFixture()
        report=probe.collect(host,python=sys.executable,system='Windows',machine='AMD64')
        self.assertEqual(host.reads,[])
        self.assertEqual([name for name,_,_ in host.commands],['python_inventory'])
        self.assertEqual(report['inventory_exit_code'],2)
        self.assertFalse(report['compatibility']['model_generation_verified'])

    def test_bad_cuda_python_data_unknown_architecture_and_model_are_redacted(self):
        host=HostFixture(); host.files['/usr/local/cuda/version.json']='{"cuda":"wrong"}'
        host.files['/proc/device-tree/model']='custom SECRET123 00:11:22:33:44:55'
        host.results['python_inventory']['text']='not json SECRET123'
        report=probe.collect(host,python=sys.executable,system='Linux',machine='SECRET123')
        self.assertEqual(report['platform']['architecture'],'unknown_redacted')
        self.assertIsNone(report['platform']['model'])
        self.assertIsNone(report['cuda_toolkit']['version_file'])
        self.assertEqual(report['python']['status'],'invalid_report')
        self.assertNotIn('SECRET123',json.dumps(report))

    def test_old_nvcc_fallback_does_not_masquerade_as_torch_cuda(self):
        host=HostFixture(); del host.files['/usr/local/cuda/version.json']
        host.results['nvcc']={'status':'completed','exit_code':0,'text':'Cuda compilation tools, release 11.4, V11.4.315'}
        report=inventory(host)
        self.assertEqual(report['cuda_toolkit']['nvcc_release'],'11.4')
        self.assertIsNone(report['cuda_toolkit']['version_file'])
        self.assertEqual(report['python']['torch_runtime']['status'],'not_requested')

    def test_runtime_cuda_failure_is_explicit_and_never_a_model_test(self):
        host=HostFixture(); data=python_fixture({'status':'import_or_query_failed','error_type':'SECRET123'})
        host.results['python_inventory']['text']=json.dumps(data)
        report=inventory(host,probe_torch=True)
        self.assertEqual(report['python']['torch_runtime'],{'status':'import_or_query_failed','blocked_actions':[]})
        self.assertEqual(report['inventory_exit_code'],2)
        self.assertIn('requested_torch_runtime_probe_incomplete',report['warnings'])
        self.assertFalse(report['python']['model_loaded']); self.assertFalse(report['python']['model_generate_called'])

    def test_runner_handles_absent_permission_timeout_and_output_bounds(self):
        host=probe.LocalHost()
        for error,status in ((FileNotFoundError('SECRET123'),'missing'),(PermissionError('SECRET123'),'permission_denied'),
                             (subprocess.TimeoutExpired('cmd',1),'timeout')):
            with patch.object(probe.subprocess,'run',side_effect=error):
                self.assertEqual(host.run('nvcc',['nvcc','--version'],1),{'status':status})
        with patch.object(probe.subprocess,'run',return_value=subprocess.CompletedProcess([],0,b'a'*(probe.MAX_BYTES+1),b'SECRET123')):
            self.assertEqual(host.run('nvcc',['nvcc','--version'],1)['status'],'output_too_large')

    def test_no_credential_environment_forwarding_and_no_shell_execution(self):
        with patch.dict(probe.os.environ,{'HF_TOKEN':'SECRET123','AWS_SECRET_ACCESS_KEY':'SECRET123','PYTHONPATH':'SECRET123'}):
            environment=probe.safe_environment()
            self.assertNotIn('HF_TOKEN',environment); self.assertNotIn('AWS_SECRET_ACCESS_KEY',environment)
            self.assertNotIn('PYTHONPATH',environment)
        with patch.object(probe.subprocess,'run',return_value=subprocess.CompletedProcess([],0,b'',b'SECRET123')) as command:
            probe.LocalHost().run('nvcc',['nvcc','--version'],3)
        self.assertFalse(command.call_args.kwargs['shell'])
        self.assertEqual(command.call_args.kwargs['timeout'],3)
        with self.assertRaises(ValueError): probe.LocalHost().run('ssh',['ssh','unknown'],1)

    def test_standalone_script_and_embedded_child_parse_with_python38_grammar(self):
        ast.parse(SCRIPT.read_text(encoding='utf-8'),feature_version=(3,8))
        ast.parse(probe.PYTHON_REPORT_CODE,feature_version=(3,8))

    def test_network_paths_are_rejected_before_path_or_interpreter_lookup(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with patch.object(probe.Path,'is_dir') as check:
                with self.assertRaises(SystemExit): probe.main(['--disk-path','//unknown/share'])
            check.assert_not_called()
            with patch.object(probe.Path,'is_dir',return_value=True),patch.object(probe.shutil,'which') as lookup:
                with self.assertRaises(SystemExit): probe.main(['--python','\\\\unknown\\python'])
            lookup.assert_not_called()

    def test_planned_resource_manifest_totals_remain_pinned_and_unapproved(self):
        manifest=json.loads((SCRIPT.parents[1]/'configs/qwen3-vl-agx-resource-plan.json').read_text(encoding='utf-8'))
        self.assertEqual(len(manifest['files']),12)
        self.assertEqual(sum(x['bytes'] for x in manifest['files']),17541458559)
        self.assertEqual(sum(x['bytes'] for x in manifest['files'] if x['kind']=='weight'),17534339512)
        self.assertEqual(manifest['planned_additional_bytes_if_existing_six_reused'],17534407540)
        self.assertFalse(manifest['generation_authorized']); self.assertFalse(manifest['downloads_performed_this_stage'])
        generation=next(x for x in manifest['files'] if x['name']=='generation_config.json')
        self.assertEqual(generation['sha256'],'8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44')
        self.assertEqual(generation['git_blob_sha1'],'e116347d7ef42621c7a5957f82c015c38d939c1f')
        self.assertTrue(generation['hash_pin_completed']); self.assertTrue(generation['hash_pin_required_before_loading'])
        self.assertFalse(generation['local_present']); self.assertFalse(generation['download_authorized'])

    def test_genuine_existing_python_metadata_probe_never_imports_a_model(self):
        result=probe.LocalHost().run('python_inventory',[sys.executable,'-I','-B','-c',probe.PYTHON_REPORT_CODE,
            '0',json.dumps(probe.PACKAGES)],15)
        self.assertEqual(result['status'],'completed')
        report=probe.parse_python(result['text'])
        self.assertEqual(report['status'],'collected')
        self.assertEqual(report['torch_runtime']['status'],'not_requested')
        self.assertFalse(report['model_loaded']); self.assertFalse(report['model_generate_called'])

    def test_child_python_audit_guard_denies_writes_sockets_and_new_processes(self):
        suffix='''
blocked={}
for name,action in [('write',lambda:open('audit-must-not-exist.txt','w')),
                    ('socket',lambda:socket.socket()),
                    ('process',lambda:__import__('subprocess').Popen(['never-execute']))]:
    try: action(); blocked[name]=False
    except RuntimeError: blocked[name]=True
print(json.dumps(blocked))
'''
        result=probe.LocalHost().run('python_inventory',[sys.executable,'-I','-B','-c',
            probe.PYTHON_REPORT_CODE+suffix,'0',json.dumps(probe.PACKAGES)],15)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(json.loads(result['text'].splitlines()[-1]),{'write':True,'socket':True,'process':True})
        self.assertFalse(Path('audit-must-not-exist.txt').exists())


if __name__=='__main__': unittest.main()
