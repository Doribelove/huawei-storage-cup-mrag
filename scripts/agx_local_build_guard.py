"""Independent task-local guard for the already-running build, not a new builder.

Only explicitly registered task sessions and their observed descendants can be
signalled. PID start ticks prevent stale PID reuse from authorizing a signal.
No services, robot processes, host settings or original deadlines are changed.
"""
import argparse
import json
import os
import signal
import stat
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.home() / 'huawei-mrag-runtime-20261007'
PROCESS_LIMIT = 2048


def process_snapshot(pid, proc_root=Path('/proc')):
    directory = proc_root / str(int(pid))
    try:
        text = (directory / 'stat').read_text()
        # comm may contain spaces or parentheses; fields begin after final ')'.
        parts = text[text.rindex(')') + 2:].split()
        if parts[0] == 'Z':
            return None
        return {'pid': int(pid), 'parent': int(parts[1]), 'group': int(parts[2]),
                'session': int(parts[3]), 'tty': int(parts[4]), 'start_ticks': int(parts[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(captured, current):
    return bool(current and all(current[key] == captured[key]
                               for key in ['pid', 'start_ticks', 'group', 'session']))


def observe_descendants(known, proc_root=Path('/proc')):
    """Walk only /proc entries already bound to the authorized task ancestry."""
    for pid in list(known):
        if not same_process(known[pid], process_snapshot(pid, proc_root)):
            del known[pid]
    queue = list(known)
    visited = set()
    while queue:
        pid = queue.pop()
        if pid in visited:
            continue
        visited.add(pid)
        if len(visited) > PROCESS_LIMIT:
            raise RuntimeError('owned process-tree bound reached')
        captured = known[pid]
        current = process_snapshot(pid, proc_root)
        if not same_process(captured, current):
            continue
        children = proc_root / str(pid) / 'task' / str(pid) / 'children'
        try:
            child_ids = children.read_text().split()
        except FileNotFoundError:
            continue
        for value in child_ids:
            child = process_snapshot(int(value), proc_root)
            if child is not None and child['parent'] == pid:
                prior = known.get(child['pid'])
                if prior and not same_process(prior, child):
                    continue
                child['task_ancestry_observed'] = True
                known[child['pid']] = child
                queue.append(child['pid'])
    return known


def task_location_matches(pid, root, proc_root=Path('/proc')):
    directory = proc_root / str(pid)
    try:
        command = (directory / 'cmdline').read_bytes()
        cwd = (directory / 'cwd').resolve()
        return (directory.stat().st_uid == os.getuid() and str(root).encode() in command
                and (cwd == root or root in cwd.parents))
    except (FileNotFoundError, ProcessLookupError):
        return False


def register_descendant_sessions(known, sessions, root, proc_root=Path('/proc')):
    for captured in known.values():
        current = process_snapshot(captured['pid'], proc_root)
        if (captured.get('task_ancestry_observed') and same_process(captured, current)
                and current['pid'] == current['group'] == current['session'] and current['tty'] == 0
                and current['session'] != getattr(os, 'getsid', lambda _: None)(0)
                and task_location_matches(current['pid'], root, proc_root)):
            sessions.add(current['session'])
    return sessions


def verified_groups(known, authorized_sessions, proc_root=Path('/proc')):
    groups = set()
    for captured in known.values():
        current = process_snapshot(captured['pid'], proc_root)
        if same_process(captured, current) and current['session'] in authorized_sessions:
            # All task subprocesses inherit their explicitly registered session.
            # Never signal this guard's own session or an unregistered session.
            if current['group'] > 1 and current['group'] != getattr(os, 'getpgrp', lambda: None)():
                groups.add(current['group'])
    return sorted(groups)


def logical_size(root):
    size = 0
    for directory, directories, files in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if not (Path(directory) / name).is_symlink()]
        for name in files:
            try:
                info = (Path(directory) / name).lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                size += info.st_size
    return size


def register_roots(root):
    rows = []
    for name, field in [('build-sequence-start', 'pid'), ('qualification-continuation-start', 'pid'),
                        ('build-stage', 'pid')]:
        path = root / 'state' / (name + '.json')
        if not path.exists():
            continue
        state = json.loads(path.read_text())
        pid = state.get(field)
        if not pid:
            continue
        row = process_snapshot(pid)
        if row is None:
            continue
        assert task_location_matches(pid, root), 'registered process owner/command/cwd does not bind to task prefix'
        assert row['group'] == row['session'] and row['tty'] == 0, 'task must already have independent session'
        assert row['session'] != os.getsid(0), 'do not register the SSH or guard session'
        rows.append({'role': name, **row})
    assert rows, 'no existing live task sessions; do not start or recreate a builder'
    return rows


def save_report(path, value):
    value['updated_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix('.json.new')
    with temporary.open('w', encoding='utf-8') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n'); output.flush(); os.fsync(output.fileno())
    os.replace(temporary, path)


def run(check_only=False):
    assert os.name == 'posix', 'actual guard requires Linux process-group semantics'
    root = ROOT.resolve()
    original = json.loads((root / 'state/build-budget.json').read_text())
    deadline = original['absolute_deadline_epoch']
    assert original['time_limit_seconds'] == 86400 and deadline > time.time()
    roots = register_roots(root)
    report = {'schema': 'independent-task-local-build-guard-v1', 'pid': os.getpid(),
              'status': 'scope_checked' if check_only else 'running', 'registered_roots': roots,
              'original_first_actual_build_started_utc': original['first_actual_build_started_utc'],
              'original_absolute_deadline_epoch': deadline, 'early_termination_epoch': deadline - 30,
              'kill_grace_seconds': 10, 'disk_early_stop_bytes': 70000000000,
              'download_limit_bytes': 25000000000, 'minimum_available_memory_bytes': 16000000000,
              'minimum_filesystem_available_bytes': 1000000000, 'build_restarted': False,
              'system_services_changed': False, 'only_registered_task_sessions_may_be_signalled': True,
              'Windows_connection_required': False}
    if check_only:
        print(json.dumps(report)); return 0
    lock = root / 'state/local-build-guard.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    state_path = root / 'state/local-build-guard.json'
    assert not state_path.exists(), 'preserve prior guard evidence; audited resume required'
    known = {row['pid']: row for row in roots}
    sessions = {row['session'] for row in roots}
    next_resource_check = 0
    reason = None
    try:
        save_report(state_path, report)
        while True:
            observe_descendants(known)
            # Native vision and qualification also use independent sessions.
            # Only observed task descendants with task-bound command/cwd/owner
            # can extend the allowlist; no unrelated /proc scan is performed.
            register_descendant_sessions(known, sessions, root)
            live_groups = verified_groups(known, sessions)
            report['verified_live_task_groups'] = live_groups
            if not live_groups:
                report['status'] = 'finished_registered_task_sessions'; break
            if time.time() >= deadline - 30:
                reason = 'original_24h_deadline_early_margin'; break
            if time.monotonic() >= next_resource_check:
                available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                                 if line.startswith('MemAvailable:'))
                filesystem = os.statvfs(root)
                free = filesystem.f_bavail * filesystem.f_frsize
                size = logical_size(root)
                charged = json.loads((root / 'state/download-budget.json').read_text())['charged_bytes']
                report.update(MemAvailable_bytes=available, logical_prefix_bytes=size,
                              filesystem_available_bytes=free, conservative_charged_download_bytes=charged)
                if available < 16000000000 or size >= 70000000000 or free < 1000000000 or charged > 25000000000:
                    reason = 'existing_task_resource_margin_reached'; break
                next_resource_check = time.monotonic() + 10
            save_report(state_path, report)
            time.sleep(2)
    except Exception as error:
        reason = 'guard_error_' + type(error).__name__
    if reason:
        report.update(status='stopping_registered_task_groups', reason=reason)
        save_report(state_path, report)
        signalled = []
        for sig in [signal.SIGTERM, signal.SIGKILL]:
            # Recheck PID start identity immediately before each signal.
            for group in verified_groups(known, sessions):
                try:
                    os.killpg(group, sig); signalled.append({'group': group, 'signal': int(sig)})
                except ProcessLookupError:
                    pass
            if sig == signal.SIGTERM:
                time.sleep(10)
        report.update(status='stopped_registered_task_groups', signalled_task_groups=signalled)
    save_report(state_path, report)
    lock.unlink()
    print(json.dumps(report))
    return 3 if reason else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    raise SystemExit(run(parser.parse_args().check_only))
