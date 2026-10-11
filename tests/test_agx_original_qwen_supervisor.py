import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name + '_supervisor_test', ROOT / 'scripts' / (name + '.py'))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


supervisor = load('agx_supervise_original_qwen')
contract = load('agx_original_qwen_contract')


class OriginalQwenSupervisorPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / 'state').mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def run_blocked(self):
        with patch.object(supervisor, 'ROOT', self.root), patch.object(supervisor, 'module', return_value=contract), \
                patch.object(supervisor.subprocess, 'Popen', side_effect=AssertionError('must not launch model')) as launch:
            with self.assertRaises((PermissionError, RuntimeError)):
                supervisor.run()
            launch.assert_not_called()
        self.assertFalse((self.root / 'state/original-Qwen-supervisor.lock').exists())

    def test_incomplete_real_gates_block_before_launch_or_new_state(self):
        (self.root / 'state/gpu-reader-qualification.json').write_text(json.dumps({'status': 'running'}))
        self.run_blocked()

    def test_elapsed_original_window_cannot_be_reset_to_launch_model(self):
        proof = {'status': 'passed', 'pretrained_weights_loaded': False, **dict.fromkeys(contract.GATE_KEYS, True)}
        (self.root / 'state/gpu-reader-qualification.json').write_text(json.dumps(proof))
        (self.root / 'state/build-budget.json').write_text(json.dumps({'absolute_deadline_epoch': 1}))
        self.run_blocked()


if __name__ == '__main__':
    unittest.main()
