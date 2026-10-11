"""Future approved single-resource transfer. Pending plans exit before any I/O."""
import argparse
import atexit
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from agx_download_budget import BudgetExceeded, BudgetLedger, confined_path, require_approval

HOSTS = {'github.com', 'codeload.github.com', 'gitlab.com', 'files.pythonhosted.org',
         'release-assets.githubusercontent.com', 'objects.githubusercontent.com',
         'www.python.org', 'python.org', 'huggingface.co', 'cas-bridge.xethub.hf.co',
         'cdn-lfs.huggingface.co', 'cdn-lfs-us-1.hf.co', 'cdn-lfs-eu-1.hf.co', 'us.aws.cdn.hf.co'}


def check_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname not in HOSTS or parsed.username or parsed.password:
        raise ValueError('resource or redirect outside official approved HTTPS hosts')


def resolve_resource_url(resource, official_redirect=None):
    check_url(resource['url'])
    if official_redirect is None:
        return resource['url']
    pinned_base = 'https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/resolve/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/'
    if (resource['phase'] != 'model' or not resource['url'].startswith(pinned_base)
            or not resource['filename'].endswith('.safetensors') or not resource.get('sha256')):
        raise ValueError('official redirect is only allowed for the fixed original weight shards')
    if len(official_redirect.encode('utf-8')) > 16384:
        raise ValueError('redirect metadata exceeds bounded input')
    check_url(official_redirect)
    if urllib.parse.urlsplit(official_redirect).hostname != 'us.aws.cdn.hf.co':
        raise ValueError('resolved model resource must use the observed official HF CDN')
    return official_redirect


class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        check_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def logical_prefix_size(prefix):
    total = 0
    for path in prefix.rglob('*'):
        if path.is_symlink():
            target = path.resolve()
            if prefix not in target.parents and target != prefix:
                raise ValueError('link outside transfer prefix; audit before continuing')
            total += path.lstat().st_size
            continue
        if path.is_file():
            total += path.stat().st_size
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--resource-id', required=True)
    parser.add_argument('--qualification')
    parser.add_argument('--resolved-url-stdin', action='store_true', help='Transient redirect from the same fixed official HF URL; never written to receipt')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    resource = next(row for row in plan['resources'] if row['id'] == args.resource_id)
    proof = json.loads(Path(args.qualification).read_text(encoding='utf-8')) if args.qualification else None
    require_approval(plan, resource['phase'], proof)
    fetch_url = resolve_resource_url(resource, sys.stdin.read(16385).strip() if args.resolved_url_stdin else None)
    if sys.platform != 'linux' or os.uname().machine != 'aarch64':
        raise ValueError('approved deployment target is Linux aarch64')
    prefix = (Path.home() / 'huawei-mrag-runtime-20261007').resolve()
    prefix.mkdir(exist_ok=True)
    state_dir = prefix / 'state'
    state_dir.mkdir(exist_ok=True)
    ledger_path = state_dir / 'download-budget.json'
    if not ledger_path.exists() and (any(prefix.glob('downloads/*')) or any(state_dir.iterdir())):
        raise ValueError('existing transfer files without ledger; do not reset budget')
    transfer_lock = state_dir / 'fetch.lock'
    descriptor = os.open(transfer_lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    atexit.register(transfer_lock.unlink)
    ledger = BudgetLedger(ledger_path, plan['limits']['download_bytes'], plan['budget']['reserved_post_gate_original_model_bytes'])
    filename = resource['filename']
    destination = confined_path(prefix, 'downloads/' + filename)
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        raise ValueError('resource already present; validate receipt instead of downloading again')
    partial = destination.with_name(destination.name + '.attempt-' + str(ledger.state['reservations']) + '.part')
    if partial.exists():
        raise ValueError('existing partial attempt must be reviewed, never overwritten')
    initial_size = logical_prefix_size(prefix)
    if initial_size >= plan['limits']['disk_bytes']:
        raise BudgetExceeded('work directory disk limit reached')
    expected = resource.get('bytes')
    if expected is not None and expected > ledger.remaining(resource['phase']):
        raise BudgetExceeded('fixed resource cannot fit remaining phase budget')
    opener = urllib.request.build_opener(OfficialRedirect())
    with opener.open(urllib.request.Request(fetch_url, headers={'User-Agent': 'Huawei-MRAG-approved-resource-fetch'}), timeout=30) as response:
        check_url(response.geturl())
        declared = response.headers.get('Content-Length')
        if declared and int(declared) > min(ledger.remaining(resource['phase']), resource['max_resource_bytes']):
            raise BudgetExceeded('declared resource exceeds allowance')
        with partial.open('xb') as output:
            written = 0
            class Writer:
                def write(self, data):
                    nonlocal written
                    if initial_size + written + len(data) + 1048576 > plan['limits']['disk_bytes']:
                        raise BudgetExceeded('download disk budget reached')
                    if written + len(data) > resource['max_resource_bytes']:
                        raise BudgetExceeded('per-resource stop threshold reached')
                    if shutil.disk_usage(prefix).free < len(data) + 1_000_000_000:
                        raise BudgetExceeded('filesystem free-space margin reached')
                    count = output.write(data)
                    written += count
                    return count
            ledger.copy(response, Writer(), resource['phase'], expected_bytes=expected)
            output.flush()
            os.fsync(output.fileno())
    sha = hashlib.sha256()
    blob = hashlib.sha1(('blob %d\0' % partial.stat().st_size).encode())
    with partial.open('rb') as stream:
        for data in iter(lambda: stream.read(1048576), b''):
            sha.update(data)
            blob.update(data)
    if resource.get('sha256') and sha.hexdigest() != resource['sha256']:
        raise ValueError('resource SHA256 mismatch; failed attempt remains charged')
    if resource.get('git_blob_sha1') and blob.hexdigest() != resource['git_blob_sha1']:
        raise ValueError('fixed JSON git blob mismatch')
    receipt = {'resource_id': resource['id'], 'source_url': resource['url'], 'bytes': partial.stat().st_size,
               'sha256': sha.hexdigest(), 'git_blob_sha1': blob.hexdigest(),
               'source_archive_first_observed_SHA256': resource.get('sha256') is None and resource['kind'] == 'source',
               'cumulative_conservative_charged_bytes': ledger.state['charged_bytes'],
               'actual_download_host': urllib.parse.urlsplit(fetch_url).hostname,
               'transient_official_redirect_used': args.resolved_url_stdin,
               'transient_redirect_URL_persisted': False}
    receipt_path = state_dir / (resource['id'] + '-receipt.json')
    if receipt_path.exists():
        raise ValueError('receipt already exists')
    with receipt_path.open('x', encoding='utf-8', newline='\n') as output:
        output.write(json.dumps(receipt, indent=2) + '\n')
    os.replace(partial, destination)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    try:
        main()
    except PermissionError as error:
        print(json.dumps({'status': 'approval_pending', 'reason': str(error)}))
        raise SystemExit(2)
    except urllib.error.HTTPError as error:
        print(json.dumps({'status': 'stopped', 'HTTP_status': error.code}))
        raise SystemExit(3)
