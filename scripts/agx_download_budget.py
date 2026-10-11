"""Persistent conservative payload budget. No network access in this module."""
import json
import os
from contextlib import contextmanager
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, path, limit=25_000_000_000, model_reserve=17_534_407_540):
        self.path = Path(path)
        self.limit = limit
        self.model_reserve = model_reserve
        if not 0 <= model_reserve <= limit:
            raise ValueError('invalid reserve')
        with self._lock():
            if self.path.exists():
                self.state = json.loads(self.path.read_text(encoding='utf-8'))
                if self.state['limit'] != limit or self.state['model_reserve'] != model_reserve:
                    raise ValueError('existing budget must not be reset or changed')
            else:
                self.state = {'schema': 1, 'limit': limit, 'model_reserve': model_reserve,
                              'charged_bytes': 0, 'bootstrap_charged_bytes': 0, 'reservations': 0}
                self._save()
        if not (0 <= self.state['bootstrap_charged_bytes'] <= self.state['charged_bytes'] <= limit):
            raise ValueError('invalid persisted counters')

    @contextmanager
    def _lock(self):
        lock_path = self.path.with_suffix(self.path.suffix + '.lock')
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        try:
            yield
        finally:
            lock_path.unlink()

    def _save(self):
        temporary = self.path.with_suffix(self.path.suffix + '.new')
        with temporary.open('w', encoding='utf-8', newline='\n') as output:
            json.dump(self.state, output, sort_keys=True)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, self.path)

    def remaining(self, phase):
        if phase not in ['bootstrap', 'model']:
            raise ValueError('unknown phase')
        remaining = self.limit - self.state['charged_bytes']
        if phase == 'bootstrap':
            remaining = min(remaining, self.limit - self.model_reserve - self.state['bootstrap_charged_bytes'])
        return remaining

    def charge_before_read(self, amount, phase):
        if amount <= 0:
            raise ValueError('positive reservation required')
        with self._lock():
            self.state = json.loads(self.path.read_text(encoding='utf-8'))
            if amount > self.remaining(phase):
                raise BudgetExceeded('cumulative payload budget reached; retries never reset it')
            self.state['charged_bytes'] += amount
            if phase == 'bootstrap':
                self.state['bootstrap_charged_bytes'] += amount
            self.state['reservations'] += 1
            self._save()

    def copy(self, reader, writer, phase, expected_bytes=None, chunk_bytes=1024 * 1024):
        if chunk_bytes <= 0 or expected_bytes is not None and expected_bytes < 0:
            raise ValueError('invalid stream length')
        received = 0
        while expected_bytes is None or received < expected_bytes:
            amount = min(chunk_bytes, self.remaining(phase))
            if expected_bytes is not None:
                amount = min(amount, expected_bytes - received)
            if not amount:
                raise BudgetExceeded('no payload allowance left')
            self.charge_before_read(amount, phase)
            data = reader.read(amount)
            if len(data) > amount:
                raise ValueError('reader exceeded requested bound')
            if not data:
                if expected_bytes is not None:
                    raise IOError('incomplete fixed-size resource')
                break
            written = writer.write(data)
            if written is not None and written != len(data):
                raise IOError('short write')
            received += len(data)
        return received


def confined_path(prefix, relative):
    prefix = Path(prefix).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('destination must stay in approved prefix')
    destination = (prefix / relative).resolve()
    if destination == prefix or prefix not in destination.parents:
        raise ValueError('destination escaped approved prefix')
    return destination


def require_approval(plan, phase, qualification=None):
    if not plan['authorization']['resource_download_and_GPU_build']:
        raise PermissionError('GPU experiment approval is pending; no network or remote write permitted')
    if phase == 'model':
        if not plan['authorization']['original_model_download_and_generation']:
            raise PermissionError('original weights require separate explicit approval')
        if not qualification or qualification.get('status') != 'passed' or not all(qualification.get(key) is True for key in
                ['CUDA_operators_passed', 'tiny_Qwen_text_passed', 'tiny_Qwen_multimodal_passed',
                 'native_processor_and_runtime_requalified']):
            raise PermissionError('real no-weight GPU gates have not passed')
