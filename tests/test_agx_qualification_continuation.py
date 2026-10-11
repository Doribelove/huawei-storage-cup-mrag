import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('agx_continuation', Path(__file__).resolve().parents[1] / 'scripts/agx_qualify_when_ready.py')
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


class QualificationContinuationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'state').mkdir()
        self.state = self.root / 'state/qualification-continuation.json'
        for name, value in [('ROOT', self.root), ('STATE', self.state), ('LOCK', self.root / 'state/qualification-continuation.lock')]:
            context = patch.object(continuation, name, value); context.start(); self.addCleanup(context.stop)
        context = patch.object(continuation.atexit, 'register'); context.start(); self.addCleanup(context.stop)
        self.write('build-budget', {'absolute_deadline_epoch': time.time() + 3600})

    def write(self, name, value):
        (self.root / 'state' / (name + '.json')).write_text(json.dumps(value))

    def test_builder_failure_never_initializes_runtime_or_starts_a_child(self):
        self.write('build-sequence', {'status': 'stopped', 'phase': 'torch', 'exit_code': 3})
        with patch.object(continuation.subprocess, 'Popen') as process, patch.object(continuation.importlib.util, 'spec_from_file_location') as loader:
            self.assertEqual(continuation.main(), 3)
            process.assert_not_called(); loader.assert_not_called()
        self.assertEqual(json.loads(self.state.read_text())['reason'], 'existing_builder_stopped')

    def test_expired_original_window_blocks_even_a_completed_builder(self):
        self.write('build-budget', {'absolute_deadline_epoch': time.time() - 1})
        self.write('build-sequence', {'status': 'completed'})
        with patch.object(continuation.subprocess, 'Popen') as process:
            self.assertEqual(continuation.main(), 3); process.assert_not_called()
        self.assertFalse((self.root / 'state/gpu-reader-qualification.json').exists())

    def test_existing_continuation_state_is_preserved_without_launch(self):
        self.state.write_bytes(b'{"status":"waiting","pid":987654}')
        before = self.state.read_bytes()
        with patch.object(continuation.subprocess, 'Popen') as process:
            with self.assertRaises(AssertionError): continuation.main()
            process.assert_not_called()
        self.assertEqual(self.state.read_bytes(), before)


if __name__ == '__main__': unittest.main()
