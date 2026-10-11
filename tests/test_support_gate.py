import json
import unittest
from evidence_lab.support import parse_support_verdict, finalize_answer
from evidence_lab.reader_processor import PinnedVisualProcessor, VisualProcessorUnavailable

class SupportGateTests(unittest.TestCase):
    def test_ambiguous_or_forged_verdict_cannot_release_candidate(self):
        candidate={'status':'not_enough_evidence','answer':'','citations':[],
            'answer_candidate_present':True,'supported_answer':False,
            'diagnostic_candidate_output':{'model_declared_output':{'status':'answered','answer':'unsupported','citations':[]}}}
        for support in ({'accepted':True},{'accepted':True,'semantic_verifier_run':True},{'accepted':False}):
            result=finalize_answer(candidate,support)
            self.assertFalse(result['supported_answer']);self.assertEqual(result['answer'],'')

    def test_duplicate_json_keys_and_non_boolean_flags_rejected(self):
        with self.assertRaises(ValueError):parse_support_verdict('{"supported":true,"supported":false}')
        with self.assertRaises(ValueError):parse_support_verdict(json.dumps({'supported':'true','all_claims_supported':True,'question_answered':True,'reason':'x'}))

    def test_uncertain_verdict_retained_as_rejection(self):
        verdict=parse_support_verdict(json.dumps({'supported':False,'all_claims_supported':False,'question_answered':False,'reason':'Cited pixels omit the numeric label'}))
        self.assertFalse(verdict['supported'])

    def test_unknown_runtime_profile_rejected_before_any_model_or_file_load(self):
        with self.assertRaises(VisualProcessorUnavailable):PinnedVisualProcessor('/missing',runtime_profile='invented')

if __name__=='__main__':unittest.main()
