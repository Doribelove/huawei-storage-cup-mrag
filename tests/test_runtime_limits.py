"""Relocation must not depend on a developer authorization file or extend a deadline."""
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from evidence_lab.runtime_limits import RuntimeLimits
from evidence_lab.safe_data import sha256_file


class RuntimeLimitTests(unittest.TestCase):
    def test_finite_standalone_budget_without_host_file(self):
        result = RuntimeLimits.resolve(max_run_seconds=600, now=1000)
        self.assertEqual(result.deadline_epoch, 1600)
        self.assertIsNone(result.authorization_deadline_epoch)
        self.assertIsNone(result.authorization_file_sha256)

    def test_explicit_host_deadline_can_only_shorten_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'authorization.json'
            for deadline, expected in ((1400, 1400), (9000, 1600)):
                path.write_text(json.dumps({'absolute_deadline_epoch': deadline}))
                limits = RuntimeLimits.resolve(max_run_seconds=600, authorization_file=path, now=1000)
                self.assertEqual(limits.deadline_epoch, expected)
                self.assertEqual(limits.authorization_file_sha256, sha256_file(path))

    def test_invalid_expired_or_missing_authorization_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'authorization.json'
            with self.assertRaises(FileNotFoundError):
                RuntimeLimits.resolve(authorization_file=path, now=1000)
            for deadline in (None, True, '9000', 0, 1000, float('nan'), float('inf')):
                path.write_text(json.dumps({'absolute_deadline_epoch': deadline}))
                with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                    RuntimeLimits.resolve(authorization_file=path, now=1000)

    def test_invalid_run_budget_is_rejected(self):
        for seconds in (True, 0, 299, 14401, 600.0, None):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                RuntimeLimits.resolve(max_run_seconds=seconds, now=1000)

    def test_batch_entry_import_never_reads_implicit_authorization(self):
        entry = Path(__file__).resolve().parents[1]/'scripts/run_agx_batch.py'
        original = Path.read_text
        def deny_implicit_authorization(path, *args, **kwargs):
            if path.name == 'project-authorization.json':
                raise AssertionError('Entry tried to load an implicit host authorization file')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', deny_implicit_authorization):
            module = runpy.run_path(str(entry), run_name='portable_batch_import_test')
        self.assertTrue(callable(module['main']))


if __name__ == '__main__':
    unittest.main()
