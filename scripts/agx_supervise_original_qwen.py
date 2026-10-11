"""Bounded local supervisor for the approved, already-acquired original model."""
import atexit
import importlib.util
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = (Path.home() / 'huawei-mrag-runtime-20261007').resolve()
STATE = ROOT / 'state/original-Qwen-supervisor.json'


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def save(value):
    value['updated_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = STATE.with_suffix('.json.new')
    with temporary.open('w', encoding='utf-8') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n'); output.flush(); os.fsync(output.fileno())
    os.replace(temporary, STATE)


def run():
    contract = module('agx_original_qwen_contract')
    proof = json.loads((ROOT / 'state/gpu-reader-qualification.json').read_text())
    contract.require_real_qualification(proof)
    deadline = json.loads((ROOT / 'state/build-budget.json').read_text())['absolute_deadline_epoch'] - 30
    if time.time() >= deadline:
        raise RuntimeError('original experiment window already reached; do not reset it')
    guard = module('agx_local_build_guard')
    stage = module('agx_gpu_build_stage')
    assert not STATE.exists() and not (ROOT / 'state/original-Qwen-diagnostic.json').exists(), 'preserve actual prior attempts'
    lock = ROOT / 'state/original-Qwen-supervisor.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor); atexit.register(lock.unlink)
    log_path = ROOT / 'logs/original-Qwen-diagnostic.txt'
    assert not log_path.exists(), 'preserve previous model log'
    report = {'schema': 'task-local-original-Qwen-supervisor-v1', 'status': 'running', 'pid': os.getpid(),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'no_weight_gates_passed': True,
              'maximum_model_execution_seconds': 7200, 'original_experiment_early_stop_epoch': deadline,
              'minimum_available_memory_bytes': 16000000000, 'directory_early_stop_bytes': 70000000000,
              'code_or_model_download_started_by_this_supervisor': False, 'system_services_changed': False}
    with log_path.open('xb', buffering=0) as output:
        child = subprocess.Popen([str(ROOT / 'env/bin/python'), '-I',
            str(ROOT / 'scripts/agx_run_original_qwen_diagnostic.py')], cwd=ROOT, env=stage.environment(),
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        captured = guard.process_snapshot(child.pid)
        assert captured and captured['pid'] == captured['group'] == captured['session']

        def stop_owned():
            for sig in [signal.SIGTERM, signal.SIGKILL]:
                if child.poll() is None and guard.same_process(captured, guard.process_snapshot(child.pid)):
                    try:
                        os.killpg(child.pid, sig)
                    except ProcessLookupError:
                        return
                    if sig == signal.SIGTERM:
                        try:
                            child.wait(timeout=10); return
                        except subprocess.TimeoutExpired:
                            pass

        atexit.register(stop_owned)
        execution_deadline = min(deadline, time.time() + 7200)
        report.update(child_pid=child.pid, child_start_ticks=captured['start_ticks'],
                      execution_deadline_epoch=execution_deadline, log='logs/' + log_path.name)
        save(report)
        try:
            next_disk_check = 0
            while child.poll() is None:
                available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                                 if line.startswith('MemAvailable:'))
                report['MemAvailable_bytes'] = available
                if time.monotonic() >= next_disk_check:
                    report['logical_prefix_bytes'] = guard.logical_size(ROOT)
                    next_disk_check = time.monotonic() + 10
                if time.time() >= execution_deadline or available < 16000000000 or report['logical_prefix_bytes'] >= 70000000000:
                    stop_owned(); report.update(status='stopped', reason='original_window_model_timeout_memory_or_disk_margin'); break
                save(report); time.sleep(2)
            if report['status'] == 'running':
                result = json.loads((ROOT / 'state/original-Qwen-diagnostic.json').read_text())
                report.update(status='completed_runtime_diagnostic' if child.returncode == 0 and
                    result.get('status') == 'completed_runtime_diagnostic' else 'stopped', child_exit=child.returncode,
                    original_weights_loaded=result.get('original_model_weights_loaded', False),
                    own_cases_executed=len(result.get('cases', [])), independent_accuracy_benchmark=False,
                    main_product_generation_enabled=False)
        except Exception as error:
            stop_owned(); report.update(status='stopped', error_type=type(error).__name__)
        finally:
            stop_owned(); atexit.unregister(stop_owned)
    save(report)
    print(json.dumps(report), flush=True)
    return 0 if report['status'] == 'completed_runtime_diagnostic' else 3


if __name__ == '__main__':
    raise SystemExit(run())
