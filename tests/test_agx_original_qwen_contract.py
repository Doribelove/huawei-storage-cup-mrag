import copy
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('original_qwen_contract_test', ROOT / 'scripts/agx_original_qwen_contract.py')
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
PROTOCOL = json.loads((ROOT / 'configs/agx-original-qwen-diagnostic.json').read_text(encoding='utf-8'))


class OriginalQwenInputAndCitationTests(unittest.TestCase):
    def test_reference_labels_are_not_in_prompt_inputs(self):
        case = copy.deepcopy(PROTOCOL['cases'][0])
        case['reference'] = {'value': 'DO-NOT-LEAK-REFERENCE-8490', 'unit': 'not in input'}
        self.assertNotIn('DO-NOT-LEAK', json.dumps(contract.input_messages(case)))
        with self.assertRaises(ValueError):
            contract.input_messages(PROTOCOL['cases'][1])
        message = contract.input_messages(PROTOCOL['cases'][1], SimpleNamespace(width=640, height=480))
        self.assertEqual(message[1]['content'][1]['type'], 'image')

    def test_every_real_gate_and_no_weight_qualification_is_required(self):
        proof = {'status': 'passed', 'pretrained_weights_loaded': False, **dict.fromkeys(contract.GATE_KEYS, True)}
        contract.require_real_qualification(proof)
        for key in ['status', 'pretrained_weights_loaded', *contract.GATE_KEYS]:
            bad = copy.deepcopy(proof)
            bad[key] = 'passed' if key != 'status' else 'running'
            with self.assertRaises(PermissionError):
                contract.require_real_qualification(bad)

    def test_exact_bound_quote_does_not_attest_semantic_support(self):
        evidence = PROTOCOL['cases'][0]['evidence']
        candidate = {'refused': False, 'value': '999', 'unit': 'V DC', 'reason': '', 'citations':
                     [{'source_id': evidence['source_id'], 'page': 1, 'kind': 'text', 'quote': '24 V DC'}]}
        result = contract.validate_candidate(json.dumps(candidate), evidence)
        self.assertTrue(result['source_binding_validity'])
        self.assertFalse(result['supported_answer'])
        self.assertEqual(result['semantic_evidence_support'], 'unverified')
        self.assertFalse(contract.compare_own_reference(result, PROTOCOL['cases'][0]['reference'])['exact_fields_match'])
        candidate['citations'][0]['quote'] = 'invented exact quote'
        self.assertFalse(contract.validate_candidate(json.dumps(candidate), evidence)['source_binding_validity'])

    def test_image_coordinates_must_bind_the_actual_original_image(self):
        evidence = PROTOCOL['cases'][1]['evidence']
        citation = {'source_id': evidence['source_id'], 'page': 1, 'kind': 'image', 'pixel_bbox': [0, 0, 640, 480]}
        value = {'refused': False, 'value': 'B', 'unit': '4', 'reason': '', 'citations': [citation]}
        self.assertTrue(contract.validate_candidate(json.dumps(value), evidence, (640, 480))['source_binding_validity'])
        for box in [[0, 0, 641, 480], [0, 0, 0, 10], [False, 0, 640, 480], [0, 0, float('inf'), 480]]:
            value['citations'][0]['pixel_bbox'] = box
            self.assertFalse(contract.validate_candidate(json.dumps(value), evidence, (640, 480))['source_binding_validity'])

    def test_refusal_withholds_candidate_values_and_rejects_wrong_source(self):
        evidence = PROTOCOL['cases'][0]['evidence']
        value = {'refused': True, 'value': '', 'unit': '', 'reason': 'No mass is stated.', 'citations': []}
        self.assertTrue(contract.validate_candidate(json.dumps(value), evidence)['source_binding_validity'])
        value['value'] = '12'
        self.assertFalse(contract.validate_candidate(json.dumps(value), evidence)['structural_validity'])
        value.update(refused=False, citations=[{'source_id': 'unseen source', 'page': 1, 'kind': 'text', 'quote': '24 V DC'}])
        self.assertFalse(contract.validate_candidate(json.dumps(value), evidence)['source_binding_validity'])

    def test_fixed_artifact_identity_and_exact_index_reject_substitutions(self):
        resource = {'id': 'qwen-00', 'filename': 'model-00001-of-00004.safetensors', 'bytes': 10, 'sha256': 'fixed'}
        resource['url'] = f'https://huggingface.co/{contract.FIXED_REPOSITORY}/resolve/{contract.FIXED_REVISION}/{resource["filename"]}'
        receipt = {'resource_id': resource['id'], 'source_url': resource['url'], 'bytes': 10, 'sha256': 'fixed'}
        observed = {'bytes': 10, 'sha256': 'fixed', 'git_blob_sha1': None}
        contract.verify_identity(resource, receipt, observed)
        with self.assertRaises(ValueError):
            contract.verify_identity(resource, receipt, {**observed, 'sha256': 'changed'})
        resources = [{'kind': 'weight', 'filename': f'fixed-{i}.safetensors'} for i in range(4)]
        mapping = {str(i): row['filename'] for i, row in enumerate(resources)}
        contract.verify_weight_index({'weight_map': mapping}, resources)
        mapping['0'] = '../other-model.safetensors'
        with self.assertRaises(ValueError):
            contract.verify_weight_index({'weight_map': mapping}, resources)


if __name__ == '__main__':
    unittest.main()
