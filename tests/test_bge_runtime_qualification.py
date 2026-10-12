import unittest
import numpy as np
from evidence_lab.semantic import DenseIndex, LocalBGE, runtime_recipe_differences

class RuntimeQualificationTests(unittest.TestCase):
    def identities(self):
        common = {'device': 'cpu', 'dtype': 'float32', 'attention': 'eager',
                  'torch_version': '2.3.1', 'threads': 4, 'loaded_safetensors_sha256': 'fixture-pinned',
                  'tokenizer_sha256': 'fixture-tokenizer', 'pooling': 'CLS', 'normalization': 'L2',
                  'max_length': 512, 'overlap_tokens': 64, 'dimension': 1024}
        return ({**common, 'device': 'cuda', 'threads': 8, 'cuda_matmul_allow_tf32': False,
                 'cuda_device_name': 'fixture-device'}, common)

    def test_cuda_index_eligibility_does_not_register_a_qualified_query(self):
        stored, query = self.identities()
        self.assertIn('device', runtime_recipe_differences(stored, query))
        index = DenseIndex.__new__(DenseIndex)
        index.identity = stored
        with self.assertRaisesRegex(ValueError, 'live numerical qualification'):
            index.search(np.ones(1024), query)

    def test_changed_embedding_recipe_is_never_a_runtime_difference(self):
        stored, query = self.identities()
        for key, value in [('loaded_safetensors_sha256', 'other'), ('tokenizer_sha256', 'other'),
                           ('pooling', 'mean'), ('normalization', 'none'), ('max_length', 256),
                           ('overlap_tokens', 32), ('dimension', 512)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'recipe differs'):
                runtime_recipe_differences({**stored, key: value}, query)

    def test_cuda_precision_and_backend_changes_remain_forbidden(self):
        stored, query = self.identities()
        for key, value in [('cuda_matmul_allow_tf32', True), ('cuda_matmul_allow_tf32', None),
                           ('cuda_device_name', ''), ('dtype', 'float16'), ('attention', 'sdpa')]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'FP32 eager'):
                runtime_recipe_differences({**stored, key: value}, query)
        with self.assertRaisesRegex(ValueError, 'recipe differs'):
            runtime_recipe_differences(query, stored)

    def test_original_cpu_torch_and_thread_qualification_route_is_preserved(self):
        _, query = self.identities()
        differences = runtime_recipe_differences({**query, 'torch_version': '2.10.0+cpu', 'threads': 8}, query)
        self.assertEqual(set(differences), {'torch_version', 'threads'})

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
