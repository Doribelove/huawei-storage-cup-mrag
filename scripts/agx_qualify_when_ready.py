"""Continue this approved run to no-weight gates after the single builder succeeds."""
import atexit
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.home() / 'huawei-mrag-runtime-20261007'
STATE = ROOT / 'state/qualification-continuation.json'
LOCK = ROOT / 'state/qualification-continuation.lock'


def save(status, **details):
    row = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'status': status,
           'pid': os.getpid(), 'native_build_started_by_this_process': False,
           'pretrained_weights_downloaded_by_this_process': False, **details}
    temporary = STATE.with_suffix('.json.new')
    with temporary.open('w') as output:
        json.dump(row, output, indent=2); output.write('\n'); output.flush(); os.fsync(output.fileno())
    os.replace(temporary, STATE)
    print(json.dumps(row), flush=True)


def stop_owned(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()


def main():
    assert not STATE.exists(), 'retain previous continuation status; audit restart'
    descriptor = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    atexit.register(LOCK.unlink)
    deadline = json.loads((ROOT / 'state/build-budget.json').read_text())['absolute_deadline_epoch'] - 30
    save('waiting_for_existing_serial_builder')
    while True:
        if time.time() >= deadline:
            save('stopped', reason='original_24h_window_reached_before_qualification'); return 3
        sequence = json.loads((ROOT / 'state/build-sequence.json').read_text())
        if sequence['status'] == 'stopped':
            save('stopped', reason='existing_builder_stopped', phase=sequence['phase'], exit_code=sequence['exit_code']); return 3
        if sequence['status'] == 'completed': break
        time.sleep(20)
    assert not (ROOT / 'state/gpu-reader-qualification.json').exists()
    assert json.loads((ROOT / 'state/vision-completed.json').read_text())['status'] == 'completed'
    spec = importlib.util.spec_from_file_location('approved_build_environment', ROOT / 'scripts/agx_gpu_build_stage.py')
    stage = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage)
    env = stage.environment()
    # CUDA device access retains the existing user's groups. Local/offline APIs
    # are fixed in the qualification script; no namespace/host settings changed.
    with (ROOT / 'logs/gpu-reader-qualification.txt').open('ab', buffering=0) as log:
        child = subprocess.Popen([str(ROOT / 'env/bin/python'), '-I', str(ROOT / 'scripts/agx_qualify_gpu_reader.py')],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        cleanup = lambda: stop_owned(child)
        atexit.register(cleanup)
        started = time.time(); execution_deadline = min(deadline, started + 1800)
        save('qualification_running', child_pid=child.pid, started_utc=datetime.now(timezone.utc).isoformat(),
             log='logs/gpu-reader-qualification.txt', maximum_qualification_seconds=1800)
        while child.poll() is None:
            available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
            if time.time() >= execution_deadline or available < 16000000000 or stage.usage() >= 70000000000:
                stop_owned(child); save('stopped', reason='qualification_time_memory_or_disk_margin'); return 3
            time.sleep(2)
        atexit.unregister(cleanup)
    proof_path = ROOT / 'state/gpu-reader-qualification.json'
    proof = json.loads(proof_path.read_text()) if proof_path.exists() else {}
    if child.returncode or proof.get('status') != 'passed':
        save('stopped', reason='actual_no_weight_gate_failed', exit_code=child.returncode,
             qualification_status=proof.get('status'), log='logs/gpu-reader-qualification.txt'); return 3
    save('passed_no_weight_gates_weights_still_not_downloaded', elapsed_seconds=time.time()-started,
         qualification='state/gpu-reader-qualification.json', weight_acquisition_route_unresolved=True)
    return 0


if __name__ == '__main__':
    def interrupt(signum, frame): raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    try: raise SystemExit(main())
    except Exception as error:
        if not STATE.exists() or json.loads(STATE.read_text()).get('pid') == os.getpid():
            save('stopped', type=type(error).__name__, reason=str(error).replace(str(Path.home()), '~'))
        else:
            print(json.dumps({'status': 'blocked', 'reason': 'existing continuation state retained'}), flush=True)
        raise SystemExit(3)
