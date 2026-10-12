"""Explicit invocation limits; host supervision remains external and authoritative."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

from .safe_data import sha256_file


@dataclass(frozen=True)
class RuntimeLimits:
    started_epoch: float
    deadline_epoch: float
    max_run_seconds: int
    authorization_deadline_epoch: float | None
    authorization_file_sha256: str | None

    @classmethod
    def resolve(cls, *, max_run_seconds=5400, authorization_file=None, now=None):
        if type(max_run_seconds) is not int or not 300 <= max_run_seconds <= 14400:
            raise ValueError('Run budget must be an integer from300 through14400seconds')
        started = time.time() if now is None else now
        if type(started) not in (int, float) or not math.isfinite(started) or started <= 0:
            raise ValueError('Invalid invocation start time')
        authorization_deadline = None
        authorization_hash = None
        if authorization_file is not None:
            path = Path(authorization_file)
            value = json.loads(path.read_text(encoding='utf-8'))
            authorization_deadline = value.get('absolute_deadline_epoch')
            if (type(authorization_deadline) not in (int, float)
                    or not math.isfinite(authorization_deadline)
                    or authorization_deadline <= started):
                raise ValueError('Authorization deadline is invalid or expired')
            authorization_hash = sha256_file(path)
        deadline = started + max_run_seconds
        if authorization_deadline is not None:
            deadline = min(deadline, authorization_deadline)
        return cls(started, deadline, max_run_seconds, authorization_deadline, authorization_hash)

    def report(self):
        return {
            'schema': 'explicit-invocation-limits-v1',
            'started_epoch': self.started_epoch,
            'deadline_epoch': self.deadline_epoch,
            'max_run_seconds': self.max_run_seconds,
            'authorization_deadline_epoch': self.authorization_deadline_epoch,
            'authorization_file_sha256': self.authorization_file_sha256,
            'host_supervisor_limits_are_not_overridden': True,
            'meaning': 'Scheduling boundary only; external supervisor enforces hard runtime and resources',
        }
