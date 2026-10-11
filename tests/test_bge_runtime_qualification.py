import unittest
import numpy as np
from evidence_lab.semantic import DenseIndex, LocalBGE

class RuntimeQualificationTests(unittest.TestCase):
    def test_unloaded_model_or_saved_pass_flags_cannot_qualify_runtime(self):
        index = DenseIndex.__new__(DenseIndex)
        model = LocalBGE.__new__(LocalBGE)
        model.identity = {'passed': True, 'torch_version': '2.3.1'}
        with self.assertRaisesRegex(ValueError, 'genuinely loaded'):
            index.qualify_query_runtime(model)

    def test_query_identity_change_requires_live_qualification(self):
        index = DenseIndex.__new__(DenseIndex)
        index.identity = {'torch_version': '2.10.0+cpu'}
        index.query_runtime_qualification = {'passed': True}
        with self.assertRaisesRegex(ValueError, 'live numerical qualification'):
            index.search(np.ones(1024), {'torch_version': '2.3.1'})

if __name__ == '__main__': unittest.main()
