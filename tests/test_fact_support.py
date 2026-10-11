import json
import unittest
from evidence_lab.fact_support import candidate_claims, support_claim_groups, parse_fact_verdict


class FactSupportTests(unittest.TestCase):
    def result(self,**changes):
        value={'claims':[{'id':0,'supported':True,'evidence':[0],'reason':'Visible source shows the same value.'}],
               'question_answered':True,'reason':'Requested field covered.'}
        value.update(changes);return value

    def test_decimal_units_and_all_clauses_preserved(self):
        claims=candidate_claims('A: 4.6 units; B: 3.7 units. The difference is 0.9 units.')
        self.assertEqual([c['claim'] for c in claims],['A: 4.6 units','B: 3.7 units','The difference is 0.9 units.'])

    def test_all_claims_and_known_evidence_required(self):
        claims=[{'id':0,'claim':'x'}]
        value,conflicts,accepted=parse_fact_verdict(json.dumps(self.result()),claims,1)
        self.assertTrue(accepted);self.assertEqual(conflicts,[])
        for change in ({'claims':[]},{'claims':[{'id':0,'supported':True,'evidence':[],'reason':'x'}]},
                       {'claims':[{'id':0,'supported':True,'evidence':[1],'reason':'x'}]},
                       {'question_answered':'true'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                parse_fact_verdict(json.dumps(self.result(**change)),claims,1)

    def test_boolean_reason_conflict_triggers_rejection_never_flip(self):
        result=self.result(claims=[{'id':0,'supported':True,'evidence':[0],'reason':'The value is not shown.'}])
        _,conflicts,accepted=parse_fact_verdict(json.dumps(result),[{'id':0,'claim':'x'}],1)
        self.assertFalse(accepted);self.assertEqual(conflicts[0]['type'],'true_flag_negative_reason')
        result['claims'][0].update(supported=False,reason='The source explicitly supports this.')
        _,conflicts,accepted=parse_fact_verdict(json.dumps(result),[{'id':0,'claim':'x'}],1)
        self.assertFalse(accepted);self.assertEqual(conflicts[0]['type'],'false_flag_positive_reason')

    def test_legacy_observation_is_a_scoped_claim_and_application_placeholder_is_not(self):
        candidate={'answer':'x','citations':[{'image_id':'A'},{'image_id':'B'}]}
        groups=support_claim_groups(candidate,{'model_declared_output':{'answer':'x','evidence_ids':['A','B']}})
        self.assertEqual(len(groups),1)
        groups=support_claim_groups(candidate,{'model_declared_output':{'citations':[{'image_id':'B','observation':'A building is shown'}]}})
        self.assertEqual(groups[1]['scope_source_index'],1)
        value=self.result(claims=[{'id':0,'supported':True,'evidence':[0],'reason':'x'},
            {'id':1,'supported':True,'evidence':[0],'reason':'x'}])
        with self.assertRaises(ValueError):parse_fact_verdict(json.dumps(value),groups,2)

    def test_false_fact_or_incomplete_requested_fields_withholds(self):
        for value in (self.result(question_answered=False),self.result(claims=[
                {'id':0,'supported':False,'evidence':[0],'reason':'Wrong period and value.'}])):
            _,_,accepted=parse_fact_verdict(json.dumps(value),[{'id':0,'claim':'x'}],1)
            self.assertFalse(accepted)


if __name__=='__main__':unittest.main()
