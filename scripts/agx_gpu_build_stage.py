"""Task-only offline extraction/build, supervised limits, resumable phase receipts."""
import argparse
import atexit
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.home() / 'huawei-mrag-runtime-20261007'
PLAN = json.loads((ROOT / 'scripts/agx-gpu-build-execution.json').read_text(encoding='utf-8'))
assert PLAN['authorization']['resource_download_and_GPU_build']
assert PLAN['limits']['download_bytes'] == 25000000000
assert PLAN['limits']['disk_bytes'] == 80000000000


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def usage():
    return sum(path.lstat().st_size for path in ROOT.rglob('*') if path.is_file() or path.is_symlink())


def write_state(path, data):
    temporary = path.with_suffix(path.suffix + '.new')
    with temporary.open('w', encoding='utf-8', newline='\n') as output:
        json.dump(data, output, indent=2)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def payload(resource):
    path = ROOT / 'downloads' / resource['filename']
    receipt = json.loads((ROOT / 'state' / (resource['id'] + '-receipt.json')).read_text())
    assert path.stat().st_size == receipt['bytes']
    assert digest(path) == receipt['sha256']
    if resource.get('sha256'):
        assert receipt['sha256'] == resource['sha256']
    return path, receipt


def extract(resource, destination):
    marker = ROOT / 'state' / ('extracted-' + hashlib.sha256(str(destination.relative_to(ROOT)).encode()).hexdigest()[:16] + '.json')
    archive_path, receipt = payload(resource)
    if marker.exists():
        previous = json.loads(marker.read_text())
        assert destination.is_dir() and previous['archive_sha256'] == receipt['sha256']
        return
    temporary = destination.with_name(destination.name + '.extracting')
    assert not temporary.exists(), 'partial extraction requires audited recovery'
    if destination.exists():
        assert destination.is_dir() and not any(destination.iterdir())
        destination.rmdir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(archive_path), mode='r:gz' if resource['kind'] != 'python' else 'r:xz') as archive:
        members = archive.getmembers()
        tops = {Path(member.name).parts[0] for member in members if Path(member.name).parts}
        assert len(tops) == 1
        top = next(iter(tops))
        required = sum(member.size for member in members if member.isfile())
        assert usage() + required + 1000000 <= PLAN['limits']['disk_bytes']
        temporary.mkdir()
        for member in members:
            relative = Path(member.name).relative_to(top)
            target = (temporary / relative).resolve()
            assert target == temporary.resolve() or temporary.resolve() in target.parents
            assert member.isdir() or member.isfile() or member.issym(), 'archive special/hardlink rejected'
            if member.issym():
                link_target = (target.parent / member.linkname).resolve()
                assert link_target == temporary.resolve() or temporary.resolve() in link_target.parents
            member.name = str(relative)
            if member.name == '.':
                continue
            archive.extract(member, path=str(temporary))
    if resource['kind'] == 'source' and resource['gitmodules_metadata']['http_status'] == 200:
        assert digest(temporary / '.gitmodules') == resource['gitmodules_metadata']['sha256']
    os.replace(temporary, destination)
    write_state(marker, {'resource_id': resource['id'], 'archive_sha256': receipt['sha256'],
                         'destination': str(destination.relative_to(ROOT)), 'logical_bytes_after': usage()})


def environment():
    result = dict(os.environ)
    for name in ['PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'LD_PRELOAD', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN']:
        result.pop(name, None)
    result.update({'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
                   'PATH': str(ROOT / 'env/bin') + ':' + str(ROOT / 'tools/cmake/bin') + ':/usr/local/cuda-11.4/bin:/usr/bin:/bin',
                   'LD_LIBRARY_PATH': str(ROOT / 'python/lib') + ':/usr/local/cuda-11.4/lib64:/usr/lib/aarch64-linux-gnu',
                   'TMPDIR': str(ROOT / 'tmp'), 'XDG_CACHE_HOME': str(ROOT / 'cache'),
                   'HF_HOME': str(ROOT / 'cache/huggingface'), 'TORCH_HOME': str(ROOT / 'cache/torch'),
                   'CUDA_CACHE_PATH': str(ROOT / 'cache/cuda'), 'TORCH_EXTENSIONS_DIR': str(ROOT / 'cache/torch_extensions'),
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
    for name in ['tmp', 'cache', 'cache/cuda']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    return result


def supervised(label, command, cwd, env, build=True):
    budget_path = ROOT / 'state/build-budget.json'
    if build:
        if budget_path.exists():
            budget = json.loads(budget_path.read_text())
        else:
            started = time.time()
            budget = {'first_actual_build_started_utc': datetime.now(timezone.utc).isoformat(),
                      'absolute_deadline_epoch': started + PLAN['limits']['build_attempt_seconds'],
                      'cumulative_active_seconds': 0.0, 'time_limit_seconds': 86400}
            write_state(budget_path, budget)
        assert time.time() < budget['absolute_deadline_epoch'] - 30, '24h build window reached'
    log_path = ROOT / 'logs' / (label + '.txt')
    env = dict(env)
    command = ['unshare', '--user', '--map-root-user', '--net', '--'] + command
    before = time.monotonic()
    phase_started_utc = datetime.now(timezone.utc).isoformat()
    last = before
    disk = usage()
    stop_reason = None
    with log_path.open('ab', buffering=0) as log:
        process = subprocess.Popen(command, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        def cleanup_owned_process():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        # Even an unexpected supervisor exception must not orphan a native build.
        atexit.register(cleanup_owned_process)
        while process.poll() is None:
            now = time.monotonic()
            if build:
                budget['cumulative_active_seconds'] += now - last
                budget['last_phase'] = label
                budget['last_update_utc'] = datetime.now(timezone.utc).isoformat()
                write_state(budget_path, budget)
                if time.time() >= budget['absolute_deadline_epoch'] - 30 or budget['cumulative_active_seconds'] >= 86370:
                    stop_reason = '24h_build_attempt_budget'
            last = now
            if int(now - before) % 5 == 0:
                disk = usage()
            available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))
            if disk >= PLAN['limits']['build_early_stop_logical_disk_bytes']:
                stop_reason = '70GB_early_disk_stop'
            if available < 16000000000:
                stop_reason = '16GB_shared_memory_margin'
            status = {'phase': label, 'status': 'running', 'started_utc': phase_started_utc,
                      'updated_utc': datetime.now(timezone.utc).isoformat(),
                      'pid': process.pid, 'active_command_seconds': now - before, 'logical_prefix_bytes': disk,
                      'MemAvailable_bytes': available, 'log': 'logs/' + log_path.name,
                      'network_namespace_isolated': True}
            write_state(ROOT / 'state/build-stage.json', status)
            if stop_reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(1)
        code = process.wait()
        atexit.unregister(cleanup_owned_process)
    if build:
        budget['cumulative_active_seconds'] += time.monotonic() - last
        write_state(budget_path, budget)
    record = {'phase': label, 'status': 'completed' if code == 0 and not stop_reason else 'stopped',
              'exit_code': code, 'stop_reason': stop_reason, 'elapsed_seconds': time.monotonic() - before,
              'logical_prefix_bytes': usage(), 'network_namespace_isolated': True, 'log': 'logs/' + log_path.name}
    write_state(ROOT / 'state/build-stage.json', record)
    assert record['logical_prefix_bytes'] <= PLAN['limits']['disk_bytes']
    if code != 0 or stop_reason:
        raise RuntimeError(json.dumps(record))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['extract', 'python', 'torch', 'vision'], required=True)
    args = parser.parse_args()
    if args.phase != 'python':
        assert json.loads((ROOT / 'state/bootstrap-fetch.json').read_text())['status'] == 'completed'
    resources = {row['id']: row for row in PLAN['resources']}
    if args.phase == 'extract':
        extract(resources['python'], ROOT / 'sources/python')
        extract(resources['cmake'], ROOT / 'tools/cmake')
        for placement in PLAN['source_placements']:
            extract(resources[placement['resource_id']], ROOT / placement['destination'])
        for path in PLAN['setup_requirements']['extra_explicit_checks']:
            assert (ROOT / 'sources/torch' / path / 'CMakeLists.txt').is_file()
        write_state(ROOT / 'state/source-extraction.json', {'status': 'completed', 'source_placements': 73,
                                                           'logical_prefix_bytes': usage()})
        return
    env = environment()
    if args.phase == 'python':
        # Python and CMake already have verified receipts; remaining torch sources
        # may download into a separate directory while these two archives build.
        extract(resources['python'], ROOT / 'sources/python')
        extract(resources['cmake'], ROOT / 'tools/cmake')
        supervised('python-configure', ['./configure', '--prefix=' + str(ROOT / 'python'), '--enable-shared',
                     '--with-ensurepip=install'], ROOT / 'sources/python', env)
        supervised('python-make', ['make', '-j2'], ROOT / 'sources/python', env)
        supervised('python-install', ['make', 'install'], ROOT / 'sources/python', env)
        interpreter = ROOT / 'python/bin/python3.12'
        supervised('python-stdlib-smoke', [str(interpreter), '-I', '-c', 'import ssl,sqlite3,ctypes,lzma,bz2,zlib;print(ssl.OPENSSL_VERSION)'], ROOT, env, build=False)
        supervised('python-venv', [str(interpreter), '-I', '-m', 'venv', str(ROOT / 'env')], ROOT, env, build=False)
        wheels = [str(payload(row)[0]) for row in PLAN['resources'] if row['kind'] == 'wheel']
        supervised('support-wheels-offline', [str(ROOT / 'env/bin/python'), '-I', '-m', 'pip', '--isolated',
                     'install', '--no-index', '--no-deps', '--cache-dir', str(ROOT / 'cache/pip'), *wheels], ROOT, env, build=False)
    else:
        env.update(PLAN['torch_process_environment'])
        if args.phase == 'vision':
            env.update(PLAN['vision_process_environment'])
        directory = ROOT / 'sources' / args.phase
        supervised(args.phase + '-wheel', [str(ROOT / 'env/bin/python'), 'setup.py', 'bdist_wheel'], directory, env)
        candidates = list((directory / 'dist').glob('*.whl'))
        assert len(candidates) == 1
        supervised(args.phase + '-install', [str(ROOT / 'env/bin/python'), '-I', '-m', 'pip', '--isolated',
                    'install', '--no-index', '--no-deps', '--cache-dir', str(ROOT / 'cache/pip'), str(candidates[0])], ROOT, env, build=False)
    write_state(ROOT / 'state' / (args.phase + '-completed.json'), {'status': 'completed', 'logical_prefix_bytes': usage()})


if __name__ == '__main__':
    def stop_supervisor(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop_supervisor)
    signal.signal(signal.SIGINT, stop_supervisor)
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'stopped', 'type': type(error).__name__,
                          'reason': str(error).replace(str(Path.home()), '~')}), flush=True)
        raise SystemExit(3)
