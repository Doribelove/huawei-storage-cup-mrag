import unittest
from evidence_lab.unit_source_reference import literal_unit_reason_conflicts,unit_atoms


class UnitReasonConsistencyTests(unittest.TestCase):
    def verdict(self,reason,supported=True):
        return {'claims':[{'id':0,'supported':supported,'evidence':[0] if supported else [],'reason':reason}]}

    def test_actual_claim_is_preserved_and_changed_value_in_true_reason_rejects(self):
        claims=[{'id':0,'claim':'Layers are 12 nm, 28 nm and 24 nm.'}]
        conflicts=literal_unit_reason_conflicts(self.verdict('Source: inner 12 nm, middle 18 nm, outer 24 nm.'),claims)
        self.assertEqual(len(conflicts),1)
        self.assertEqual(conflicts[0]['literal_claim_numeric_unit_mentions'],[['12','nm'],['28','nm'],['24','nm']])
        self.assertFalse(conflicts[0]['source_facts_established'])
        self.assertEqual(claims[0]['claim'],'Layers are 12 nm, 28 nm and 24 nm.')

    def test_correct_values_and_decimal_equivalents_do_not_reject(self):
        claims=[{'id':0,'claim':'Layers are 12 nm, 18 nm and 24 nm.'}]
        self.assertEqual(literal_unit_reason_conflicts(self.verdict('Source: 12.0 nm, 18 nm, 24 nm.'),claims),[])

    def test_short_reasons_extra_source_values_and_mixed_units_are_not_false_conflicts(self):
        claims=[{'id':0,'claim':'Layers are 12 nm, 18 nm and 24 nm.'}]
        for reason in ('Middle layer is 18 nm.','12 nm, 18 nm, 24 nm; prior source 28 nm.',
                       'Measured values are 12 mm, 18 mm, 24 mm.'):
            self.assertEqual(literal_unit_reason_conflicts(self.verdict(reason),claims),[])

    def test_false_verdict_is_never_turned_true_by_matching_source_reason(self):
        claims=[{'id':0,'claim':'Layers are 12 nm, 18 nm and 24 nm.'}]
        verdict=self.verdict('12 nm, 18 nm, 24 nm.',False)
        self.assertEqual(literal_unit_reason_conflicts(verdict,claims),[])
        self.assertFalse(verdict['claims'][0]['supported'])

    def test_scientific_values_are_whole_literals_and_decimal_agreement_is_exact(self):
        self.assertEqual(unit_atoms('1e-3 V and 2e-3 V.',['V']),[('1e-3','V'),('2e-3','V')])
        claims=[{'id':0,'claim':'Signals are 1e-3 V and 2e-3 V.'}]
        self.assertEqual(literal_unit_reason_conflicts(self.verdict('Source: 0.001 V and 0.002 V.'),claims),[])
        self.assertEqual(len(literal_unit_reason_conflicts(self.verdict('Source: 0.001 V and 0.003 V.'),claims)),1)


if __name__=='__main__':unittest.main()
