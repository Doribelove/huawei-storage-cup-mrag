import copy
import unittest
from test_confusion_fields import fixture, catalog
from evidence_lab.source_calculation import requested_matrix_candidate, compose_source_calculated_candidate


class SourceCalculationTests(unittest.TestCase):
    def test_reference_pairs_counts_and_both_denominators(self):
        sources = catalog(fixture())
        original = copy.deepcopy(sources)
        candidate = requested_matrix_candidate('Compare misclassification rates for A, B, and C', sources)
        self.assertIn('Reference A was predicted B in 1 cases and predicted C in 4 cases', candidate['answer'])
        self.assertIn('5/47 = 10.64%', candidate['answer'])
        self.assertIn('A 8/50 = 16.00%', candidate['answer'])
        self.assertEqual(sources, original)
        self.assertFalse(candidate['semantic_support_established'])

    def test_reversed_axes_change_reference_destinations(self):
        candidate = requested_matrix_candidate('Compare misclassification rates for A B C', catalog(fixture(reverse_axes=True)))
        self.assertIn('Reference A was predicted B in 3 cases and predicted C in 5 cases', candidate['answer'])
        self.assertIn('8/50 = 16.00%', candidate['answer'])

    def test_permuted_columns_follow_labels(self):
        regular = requested_matrix_candidate('Misclassification rates for A B C', catalog(fixture()))
        permuted = requested_matrix_candidate('Misclassification rates for A B C', catalog(fixture(('B','A','C'))))
        self.assertEqual(regular['answer'], permuted['answer'])

    def test_missing_question_classes_and_metric_substitution_abstain(self):
        for question in ('Misclassification rates for A B', 'Misclassification rates',
                         'Misclassification rates for A B C using F1', 'List parameters for A B C'):
            self.assertIsNone(requested_matrix_candidate(question, catalog(fixture())))

    def test_two_source_matrices_are_ambiguous(self):
        sources = catalog(fixture())
        sources['another'] = copy.deepcopy(next(iter(sources.values())))
        self.assertIsNone(requested_matrix_candidate('Misclassification rates for A B C', sources))

    def test_unregistered_context_cannot_compose(self):
        with self.assertRaises(ValueError):
            compose_source_calculated_candidate('Misclassification rates for A B C', {}, object())


if __name__=='__main__':
    unittest.main()
