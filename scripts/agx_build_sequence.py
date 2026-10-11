"""One detached controller; source extraction, torch and vision run serially."""
import atexit
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.home() / 'huawei-mrag-runtime-20261007'
assert json.loads((ROOT / 'state/bootstrap-fetch.json').read_text())['status'] == 'completed'
assert json.loads((ROOT / 'state/python-completed.json').read_text())['status'] == 'completed'
lock = ROOT / 'state/build-sequence.lock'
descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
os.write(descriptor, str(os.getpid()).encode())
os.close(descriptor)
atexit.register(lock.unlink)
completed = []
for phase in ['extract', 'torch', 'vision']:
    marker = ROOT / 'state' / ('source-extraction.json' if phase == 'extract' else phase + '-completed.json')
    if marker.exists():
        assert json.loads(marker.read_text())['status'] == 'completed'
        completed.append(phase)
        continue
    status = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'phase': phase, 'controller_pid': os.getpid(), 'completed': completed}
    (ROOT / 'state/build-sequence.json').write_text(json.dumps(status, indent=2) + '\n')
    result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/agx_gpu_build_stage.py'), '--phase', phase],
                            cwd=str(ROOT), env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    if result.returncode:
        status.update(status='stopped', exit_code=result.returncode, stopped_utc=datetime.now(timezone.utc).isoformat())
        (ROOT / 'state/build-sequence.json').write_text(json.dumps(status, indent=2) + '\n')
        raise SystemExit(result.returncode)
    completed.append(phase)
(ROOT / 'state/build-sequence.json').write_text(json.dumps({'status': 'completed', 'completed': completed,
    'completed_utc': datetime.now(timezone.utc).isoformat(), 'weights_downloaded': False}, indent=2) + '\n')
