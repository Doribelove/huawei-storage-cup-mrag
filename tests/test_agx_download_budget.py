import io
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from agx_download_budget import BudgetExceeded, BudgetLedger, confined_path, require_approval


class AGXBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / 'ledger.json'

    def tearDown(self):
        self.temporary.cleanup()

    def test_charge_persisted_before_any_reader_effect(self):
        ledger = BudgetLedger(self.path, limit=10, model_reserve=4)
        class Reader:
            def read(_, n):
                self.assertEqual(json.loads(self.path.read_text())['charged_bytes'], n)
                return b'a' * n
        self.assertEqual(ledger.copy(Reader(), io.BytesIO(), 'bootstrap', expected_bytes=3), 3)

    def test_model_reserve_survives_bootstrap(self):
        ledger = BudgetLedger(self.path, 10, 4)
        ledger.charge_before_read(6, 'bootstrap')
        with self.assertRaises(BudgetExceeded):
            ledger.charge_before_read(1, 'bootstrap')
        ledger.charge_before_read(4, 'model')
        with self.assertRaises(BudgetExceeded):
            ledger.charge_before_read(1, 'model')

    def test_restarts_and_retries_never_refund(self):
        ledger = BudgetLedger(self.path, 10, 4)
        class Failure:
            def read(_, n):
                raise IOError('interrupted')
        with self.assertRaises(IOError):
            ledger.copy(Failure(), io.BytesIO(), 'bootstrap', expected_bytes=4)
        reopened = BudgetLedger(self.path, 10, 4)
        self.assertEqual(reopened.state['charged_bytes'], 4)
        with self.assertRaises(BudgetExceeded):
            reopened.charge_before_read(3, 'bootstrap')

    def test_unknown_length_stream_bounded(self):
        ledger = BudgetLedger(self.path, 10, 4)
        output = io.BytesIO()
        with self.assertRaises(BudgetExceeded):
            ledger.copy(io.BytesIO(b'abcdefghij'), output, 'bootstrap', chunk_bytes=2)
        self.assertEqual(output.getvalue(), b'abcdef')
        self.assertEqual(ledger.state['charged_bytes'], 6)

    def test_incomplete_resource_is_counted_and_rejected(self):
        ledger = BudgetLedger(self.path, 10, 4)
        with self.assertRaises(IOError):
            ledger.copy(io.BytesIO(b'ab'), io.BytesIO(), 'bootstrap', expected_bytes=4, chunk_bytes=2)
        self.assertEqual(ledger.state['charged_bytes'], 4)

    def test_existing_budget_cannot_change(self):
        BudgetLedger(self.path, 10, 4)
        with self.assertRaises(ValueError):
            BudgetLedger(self.path, 20, 4)

    def test_two_handles_refresh_before_charging(self):
        first = BudgetLedger(self.path, 10, 4)
        second = BudgetLedger(self.path, 10, 4)
        first.charge_before_read(4, 'bootstrap')
        with self.assertRaises(BudgetExceeded):
            second.charge_before_read(3, 'bootstrap')
        self.assertEqual(json.loads(self.path.read_text())['charged_bytes'], 4)

    def test_stale_lock_blocks_instead_of_restarting_budget(self):
        BudgetLedger(self.path, 10, 4)
        self.path.with_suffix('.json.lock').write_bytes(b'')
        with self.assertRaises(FileExistsError):
            BudgetLedger(self.path, 10, 4)

    def test_pending_approval_blocks_all_phases(self):
        plan = {'authorization': {'resource_download_and_GPU_build': False,
                                  'original_model_download_and_generation': False}}
        for phase in ['bootstrap', 'model']:
            with self.assertRaises(PermissionError):
                require_approval(plan, phase)

    def test_weights_need_both_explicit_approval_and_all_real_gates(self):
        plan = {'authorization': {'resource_download_and_GPU_build': True,
                                  'original_model_download_and_generation': True}}
        with self.assertRaises(PermissionError):
            require_approval(plan, 'model', {'CUDA_operators_passed': True})
        proof = {key: True for key in ['CUDA_operators_passed', 'tiny_Qwen_text_passed',
                                      'tiny_Qwen_multimodal_passed', 'native_processor_and_runtime_requalified']}
        with self.assertRaises(PermissionError):
            require_approval(plan, 'model', dict(proof, status='stopped'))
        proof['status'] = 'passed'
        require_approval(plan, 'model', proof)

    def test_destination_traversal_and_prefix_are_rejected(self):
        for relative in ['../outside', str(self.root.resolve()), '.']:
            with self.assertRaises(ValueError):
                confined_path(self.root, relative)
        self.assertEqual(confined_path(self.root, 'downloads/file'), self.root / 'downloads/file')


if __name__ == '__main__':
    unittest.main()
