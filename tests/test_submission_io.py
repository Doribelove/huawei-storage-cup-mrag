"""No denominator loss, withheld-answer leak, or invented exported source metadata."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from evidence_lab.safe_data import sha256_file
from evidence_lab.submission_io import configure_inventory, export_final_answers, final_citation_sources


class SubmissionIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, value):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def batch(self):
        questions = [{'sample_id': name, 'question': 'What is shown?'} for name in ('positive', 'withheld', 'error')]
        cases = []
        for question, status in zip(questions, ('answered', 'not_enough_evidence', 'failed')):
            released = status == 'answered'
            row = {**question, 'completed': True, 'status': status, 'supported_answer': released,
                   'answer': 'Fixture answer' if released else '',
                   'citations': [{'image_id': 'fixture-image'}] if released else [],
                   'citation_sources': [{'image_id': 'fixture-image', 'document_name': 'fixture-doc', 'page': 2}] if released else []}
            path = self.write(question['sample_id']+'/result.json', row)
            cases.append({'sample_id': question['sample_id'], 'result_file': str(path)})
        self.write('summary.json', {'question_denominator': 3, 'completed_attempts': 3, 'cases': cases})
        return questions, cases

    def test_refusal_and_failure_rows_remain_in_export_order(self):
        questions, _ = self.batch()
        output = export_final_answers(questions, self.root)
        self.assertEqual(output['question_count'], 3)
        self.assertEqual([r['sample_id'] for r in output['results']], [q['sample_id'] for q in questions])
        self.assertEqual([r['answer'] for r in output['results']], ['Fixture answer', '', ''])
        self.assertFalse(output['official_schema_confirmed'])

    def test_withheld_candidate_text_cannot_become_final_output(self):
        questions, _ = self.batch()
        path = self.root/'withheld/result.json'
        row = json.loads(path.read_text())
        row['answer'] = 'Unverified candidate'
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'Withheld'):
            export_final_answers(questions, self.root)

    def test_incomplete_summary_cannot_reduce_denominator(self):
        questions, cases = self.batch()
        self.write('summary.json', {'question_denominator': 3, 'completed_attempts': 2, 'cases': cases[:2]})
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            export_final_answers(questions, self.root)

    def test_wrong_final_page_catalog_is_rejected(self):
        questions, _ = self.batch()
        path = self.root/'positive/result.json'
        row = json.loads(path.read_text())
        row['citation_sources'][0]['image_id'] = 'unrelated-image'
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
            export_final_answers(questions, self.root)

    def test_export_cannot_borrow_another_batch_result(self):
        questions, cases = self.batch()
        cases[0]['result_file'] = str(self.root.parent/'other-batch/result.json')
        self.write('summary.json', {'question_denominator': 3, 'completed_attempts': 3, 'cases': cases})
        with self.assertRaisesRegex(ValueError, 'outside this batch'):
            export_final_answers(questions, self.root)

    def test_source_fields_come_from_registered_catalog_without_an_answer_quote(self):
        source = {'image_id': 'fixture-image', 'document_name': 'fixture-doc', 'page': 3,
                  'doc_sha256': 'a'*64, 'pdf_member': 'docs/example.pdf', 'parent_evidence_id': 'b'*64,
                  'image': {'sha256': 'c'*64}}
        citation = {'image_id': 'fixture-image', 'region': {'bbox_half_open': [0, 0, 4, 5]}}
        output = final_citation_sources([citation], {'fixture-image': source})
        self.assertEqual(output[0]['page'], 3)
        self.assertEqual(output[0]['pdf_sha256'], 'a'*64)
        self.assertEqual(output[0]['region'], citation['region'])
        self.assertNotIn('quote', output[0])
        with self.assertRaisesRegex(ValueError, 'matching verified'):
            final_citation_sources([citation], {})

    def test_text_quote_export_preserves_literal_comparator_without_inventing_coordinates(self):
        source = {'image_id': 'fixture-image', 'document_name': 'fixture-doc', 'page': 3,
                  'doc_sha256': 'a'*64, 'pdf_member': 'docs/example.pdf', 'parent_evidence_id': 'b'*64,
                  'image': {'sha256': 'c'*64}}
        citation = {'type': 'text_quote', 'image_id': 'fixture-image',
                    'quote': 'At least two samples (\u22652) are required.'}
        result = final_citation_sources([citation], {'fixture-image': source})[0]
        self.assertEqual(result['quote'], citation['quote'])
        self.assertNotIn('region', result)
        self.assertEqual(result['pdf_sha256'], source['doc_sha256'])

    def inventory(self):
        self.write('data/audit/data-audit.json', {'fixture': True})
        self.write('data/full/regions.jsonl', {'evidence_id': 'fixture-image-region', 'type': 'image'})
        self.write('inventory/features.jsonl', {'evidence_id': 'fixture-image-region', 'status': 'failed'})
        summary = {'schema': 'question-independent-native-image-feature-inventory-v1', 'image_denominator': 1,
                   'completed_attempts': 1, 'corpus_sha256': sha256_file(self.root/'data/full/regions.jsonl'),
                   'original_audit_sha256': sha256_file(self.root/'data/audit/data-audit.json'),
                   'features_sha256': sha256_file(self.root/'inventory/features.jsonl')}
        self.write('inventory/summary.json', summary)
        template = {'native_image_features_sha256': 'old', 'max_new_tokens': 512, 'fact_support': True}
        path = self.write('config.json', template)
        return path, template, summary

    def test_inventory_rebinding_changes_only_actual_artifact_hash_and_keeps_failures(self):
        path, template, summary = self.inventory()
        actual = configure_inventory(path, self.root, self.root/'inventory')
        expected = copy.deepcopy(template)
        expected['native_image_features_sha256'] = summary['features_sha256']
        self.assertEqual(actual, expected)

    def test_changed_source_bytes_invalidate_inventory_binding(self):
        path, _, _ = self.inventory()
        self.write('data/full/regions.jsonl', {'evidence_id': 'different-image', 'type': 'image'})
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            configure_inventory(path, self.root, self.root/'inventory')

    def test_recomputed_digest_cannot_hide_an_omitted_image(self):
        path, _, summary = self.inventory()
        self.write('inventory/features.jsonl', {'evidence_id': 'unrelated-image', 'status': 'collected'})
        summary['features_sha256'] = sha256_file(self.root/'inventory/features.jsonl')
        self.write('inventory/summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'every canonical source image'):
            configure_inventory(path, self.root, self.root/'inventory')


if __name__ == '__main__':
    unittest.main()
