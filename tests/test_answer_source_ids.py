import copy
import json
import tempfile
import unittest
from evidence_lab.reader import bind_answer_source_ids, VisualOutputInvalid, ReaderUnavailable, prepare_reader_visual_inputs
from evidence_lab.support import finalize_answer
from test_reader_visual_input import prepare_fixture


class AnswerSourceIdTests(unittest.TestCase):
    def test_references_bind_without_any_semantic_release_or_model_observation(self):
        with tempfile.TemporaryDirectory() as root:
            prepared,hits,_,kwargs=prepare_fixture(root)
            context=prepare_reader_visual_inputs('Which source is visible?',hits,prompt_projection='answer_source_ids_v3',**kwargs)
            source=next(iter(context.citation_catalog))
            raw={'answer':'An unverified answer.','evidence_ids':[source,source]}
            result=bind_answer_source_ids(json.dumps(raw),context)
            self.assertEqual(result['diagnostic_candidate_output']['model_declared_output'],raw)
            effective=result['diagnostic_candidate_output']['effective_candidate_output']
            self.assertEqual(effective['answer'],raw['answer'])
            self.assertEqual(len(effective['citations']),1)
            self.assertEqual(result['answer_source_id_projection']['repeated_reference_count'],1)
            self.assertFalse(result['supported_answer'])
            self.assertEqual(result['answer'],'')
            binding=result['diagnostic_candidate_output']['citation_bindings'][0]
            self.assertFalse(binding['observation_declared_by_model'])
            self.assertFalse(binding['observation_is_source_fact'])
            self.assertFalse(finalize_answer(result,{'accepted':True})['supported_answer'])
            self.assertNotIn('observation',context.messages[0]['content'].split('Return ONLY')[1].split('Cover all')[0].lower().replace('no observations',''))

    def test_unknown_ids_status_observations_flags_and_truncation_still_fail(self):
        with tempfile.TemporaryDirectory() as root:
            prepared,_,_,_=prepare_fixture(root)
            source=next(iter(prepared.citation_catalog))
            cases=[{'answer':'x','evidence_ids':['unknown']},{'answer':'x','evidence_ids':[]},
                   {'answer':'','evidence_ids':[source]},{'answer':' ','evidence_ids':[]},
                   {'answer':'x','evidence_ids':[True]},{'answer':'x','evidence_ids':[source],'status':'answered'},
                   {'answer':'x','evidence_ids':[source],'supported':True}]
            for item in cases:
                with self.subTest(item=item),self.assertRaises(VisualOutputInvalid):
                    bind_answer_source_ids(json.dumps(item),prepared)
            for raw in ('{"answer":"x","evidence_ids":[','{"answer":"x","answer":"y","evidence_ids":[]}'):
                with self.assertRaises(VisualOutputInvalid):bind_answer_source_ids(raw,prepared)
            result=bind_answer_source_ids('{"answer":"","evidence_ids":[]}',prepared)
            self.assertFalse(result['answer_candidate_present'])

    def test_live_packet_tampering_rejects_before_binding(self):
        with tempfile.TemporaryDirectory() as root:
            prepared,_,_,_=prepare_fixture(root)
            source=next(iter(prepared.citation_catalog))
            prepared.citation_catalog[source]['page']+=1
            with self.assertRaises(ReaderUnavailable):
                bind_answer_source_ids(json.dumps({'answer':'x','evidence_ids':[source]}),prepared)


if __name__=='__main__':unittest.main()
