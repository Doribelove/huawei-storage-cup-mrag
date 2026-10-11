"""Approved resumable bootstrap fetch; never fetches gated pretrained weights."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.home() / 'huawei-mrag-runtime-20261007'
PLAN = ROOT / 'scripts/agx-gpu-build-execution.json'
STATE = ROOT / 'state/bootstrap-fetch.json'


def save(status, **extra):
    ledger = json.loads((ROOT / 'state/download-budget.json').read_text())
    report = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'status': status,
              'completed_count': len(completed), 'completed_resource_ids': completed,
              'charged_download_bytes': ledger['charged_bytes'], 'GPU_build_started': False,
              'pretrained_weights_downloaded': False, **extra}
    temporary = STATE.with_suffix('.json.new')
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, STATE)
    print(json.dumps(report), flush=True)


plan = json.loads(PLAN.read_text(encoding='utf-8'))
assert plan['authorization']['resource_download_and_GPU_build']
completed = []
resources = [row for row in plan['resources'] if row['phase'] == 'bootstrap']
order = {'python': 0, 'cmake': 1, 'wheel': 2, 'Qwen_JSON': 3, 'source': 4}
resources.sort(key=lambda row: (order[row['kind']], row['id']))
save('running', resource_count=len(resources))
for resource in resources:
    receipt = ROOT / 'state' / (resource['id'] + '-receipt.json')
    if receipt.exists():
        data = json.loads(receipt.read_text(encoding='utf-8'))
        payload = ROOT / 'downloads' / resource['filename']
        if not payload.is_file() or payload.stat().st_size != data['bytes']:
            save('stopped', reason='receipt_without_matching_payload', resource_id=resource['id'])
            raise SystemExit(3)
        completed.append(resource['id'])
        continue
    save('running', current_resource_id=resource['id'], current_resource_kind=resource['kind'])
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/agx_bounded_fetch.py'),
                             '--plan', str(PLAN), '--resource-id', resource['id']],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ,
                            'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'})
    log = ROOT / 'logs' / (resource['id'] + '-fetch.txt')
    log.write_bytes(result.stdout + b'\n' + result.stderr)
    if result.returncode:
        save('stopped', reason='bounded_resource_fetch_failed', resource_id=resource['id'],
             exit_code=result.returncode, log='logs/' + log.name)
        raise SystemExit(result.returncode)
    completed.append(resource['id'])
    save('running', current_resource_id=resource['id'], last_resource_completed=True)
save('completed', resource_count=len(resources))
