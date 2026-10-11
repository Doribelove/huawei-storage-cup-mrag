import unittest
from evidence_lab.semantic_sanity import metric_sanity

class MetricSanityTests(unittest.TestCase):
    def test_f1_complement_is_not_misclassification_rate(self):
        for answer in ['Misclassification rates are based on the F1-scores.',
                       'The misclassification rate is calculated as1-F1.']:
            self.assertFalse(metric_sanity('What are the misclassification rates?',{'answer':answer})['passed'])

    def test_reference_to_both_metrics_does_not_establish_an_invalid_substitution(self):
        output={'answer':'Misclassification is off-diagonal count divided by the column total. The separately reported F1-score is0.98.'}
        self.assertTrue(metric_sanity('Compare misclassification rates.',output)['passed'])
        self.assertTrue(metric_sanity('What is the F1-score?',{'answer':'F1=.98'})['passed'])
        self.assertFalse(metric_sanity('What is the F1-score?',{'answer':'F1=.98'})['establishes_support'])

if __name__ == '__main__':unittest.main()
