import copy
import json
import unittest

from test_parameter_fields import catalog
from evidence_lab.parameter_fields import requested_parameter_candidate
from evidence_lab.parameter_support import parameter_source_projection
from evidence_lab.fact_support import source_support_declaration, support_messages, candidate_claims


class ParameterSupportTests(unittest.TestCase):
    def fixture(self):
        source = catalog()
        source['image-diagnostic']['document_name'] = 'synthetic'
        words = source['image-diagnostic']['readable_viewport']['native_word_records']
        for text, x in (('Median', 40), ('units:', 90), ('arbitrary', 140), ('2031', 200)):
            words.append({'text': text, 'x0': x, 'x1': x + 30, 'top': 70, 'bottom': 80})
        question = 'Give δ2 and θ2 measurements.'
        literal = requested_parameter_candidate(question, source)
        return question, source, {'source_literal_candidate': literal}

    def test_all_requested_values_titles_preface_and_columns_retained_without_mutation(self):
        question, source, validated = self.fixture()
        before = copy.deepcopy(source)
        projected = parameter_source_projection(question, validated, source)['image-diagnostic']
        self.assertIn('Table 9: Source Alpha', projected['native_table_heading_and_preface_lines'])
        self.assertIn('Median units: arbitrary 2031', projected['native_table_heading_and_preface_lines'])
        for row, actual in zip(projected['native_parameter_rows'], source['image-diagnostic']['readable_viewport']['native_parameter_rows']):
            self.assertEqual(row, {key: actual[key] for key in ('row_label', 'columns', 'values')})
        self.assertFalse(projected['native_parameter_projection']['semantic_support_established'])
        self.assertTrue(projected['native_parameter_projection']['complete_source_image_retained'])
        self.assertEqual(source, before)

    def test_tampered_candidate_word_value_title_or_source_identity_is_rejected(self):
        for mutation in ('answer', 'plan', 'native_value', 'image', 'document'):
            question, source, validated = self.fixture()
            if mutation == 'answer':
                validated['source_literal_candidate']['answer'] += ' invented'
            elif mutation == 'plan':
                validated['source_literal_candidate']['source_field_plan']['groups'][0]['title'] = 'Fabricated title'
            elif mutation == 'native_value':
                source['image-diagnostic']['readable_viewport']['native_word_records'][-7]['text'] = 'altered'
            elif mutation == 'image':
                source['image-diagnostic']['image']['sha256'] = 'e' * 64
            else:
                source['image-diagnostic']['doc_sha256'] = 'f' * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                parameter_source_projection(question, validated, source)

    def test_original_model_only_candidate_keeps_all_legacy_parameter_rows(self):
        question, source, _ = self.fixture()
        self.assertIsNone(parameter_source_projection(question, {}, source))
        citation = {'image_id': 'image-diagnostic', 'region': {'bbox_half_open': [0, 0, 100, 100]}}
        declared = source_support_declaration(0, citation, source['image-diagnostic'], question, 'wrong candidate')
        self.assertNotIn('native_parameter_projection', declared)
        self.assertEqual(declared['native_parameter_rows'], [
            {key: row[key] for key in ('row_label', 'columns', 'values')}
            for row in source['image-diagnostic']['readable_viewport']['native_parameter_rows']])

    def test_actual_message_builder_preserves_each_literal_claim_and_bounded_reread(self):
        question, _, validated = self.fixture()
        claims = candidate_claims(validated['source_literal_candidate']['answer'])
        content = [{'type': 'image', 'image': 'original.png'}]
        previous = {'fact_verdict': {'claims': [{'id': c['id'], 'supported': False} for c in claims]}}
        messages = support_messages(question, claims, {}, content, pass_index=1, previous=previous)
        payload = json.loads(messages[1]['content'][0]['text'])
        self.assertEqual(payload['claim_groups'], claims)
        self.assertEqual(payload['reread_audit']['previous_rejected_claim_ids'], [c['id'] for c in claims])
        self.assertEqual(messages[1]['content'][1:], content)
        self.assertIn('only fresh reread', messages[0]['content'][0]['text'])


if __name__ == '__main__':
    unittest.main()
