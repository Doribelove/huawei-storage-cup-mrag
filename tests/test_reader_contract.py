"""Reader schema/provenance fixtures, not a running Qwen or accuracy evaluation."""
import json
from pathlib import Path
import tempfile
import unittest
from evidence_lab.reader import LocalQwenReader,ReaderUnavailable,reader_preflight,validate_reader_output,prepare_reader_sources
from test_pilot import fixture


class ReaderContractTests(unittest.TestCase):
    def test_reader_source_boundary_is_unique_bounded_and_answer_isolated(self):
        row=fixture('a',0,text='Original source only')
        clean=prepare_reader_sources([{**row,'score':1,'unknown_metadata':'ignored'}])
        self.assertNotIn('score',clean[0]);self.assertNotIn('unknown_metadata',clean[0])
        for rows in [[],[row,row],[{**row,'gt_answer':'synthetic_label'}],
                     [{**row,'metadata':{'answer':'synthetic_label'}}],
                     [{**row,'score':'label'}],[{**row,'score':float('nan')}],
                     [{**row,'score':True}],[{**row,'route_ranks':[{'score':1}]}],
                     [{**row,'role':'historical_image_analysis'}],
                     [fixture(str(i),i) for i in range(14)],
                     [{**row,'text':'x'*12001}],['not_a_record']]:
            with self.assertRaises(ValueError):prepare_reader_sources(rows)
        self.assertEqual(prepare_reader_sources([{**row,'text':'x'*12000}])[0]['text'],'x'*12000)

    def test_invalid_context_is_rejected_before_any_model_processor_or_generation(self):
        reader=LocalQwenReader.__new__(LocalQwenReader)
        row=fixture('a',0)
        for rows in [[row,row],[{**row,'gt_answer':'synthetic_label'}]]:
            with self.assertRaises(ValueError):reader.read('Question',rows)
        # Valid but over-capacity source records are omitted whole; missing tokens close the gate.
        refused=reader.read('Question',[{**row,'text':'x'*12001}])
        self.assertFalse(refused['answer_generated'])
        self.assertFalse(refused['actual_reader_run'])
        self.assertEqual(refused['context_assembly']['omitted'][0]['reasons'],['character_budget'])
        for budget in [{'max_new_tokens':True},{'max_new_tokens':256.5},{'max_new_tokens':'256'},
                       {'max_seconds':True},{'max_seconds':float('nan')},{'max_seconds':float('inf')}]:
            with self.assertRaises(ValueError):reader.read('Question',[row],**budget)

    def test_missing_real_weights_is_a_blocker_not_a_mock_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            report=reader_preflight(Path(folder))
            self.assertFalse(report['ready'])
            self.assertIn('model-00001-of-00004.safetensors', report['missing_files'])
            self.assertIn('preprocessor_config.json', report['missing_files'])
            self.assertIn('video_preprocessor_config.json', report['missing_files'])
            with self.assertRaises(ReaderUnavailable): LocalQwenReader(Path(folder))

    def test_exact_quote_binding_never_claims_semantic_correctness(self):
        row=fixture('a',0,text='Receiving Clerk maintains the diagnostic report.')
        value={'status':'answered','answer':'A generated candidate answer',
               'citations':[{'evidence_id':'a','quote':'Receiving Clerk'}]}
        result=validate_reader_output(json.dumps(value),[row])
        self.assertTrue(result['quote_bindings_all_passed']);self.assertTrue(result['answer_generated'])
        self.assertFalse(result['supported_answer'])
        self.assertEqual(result['semantic_correctness'],'not_established_by_exact_quote_binding')
        for bad in [{**value,'citations':[{'evidence_id':'foreign','quote':'Receiving Clerk'}]},
                    {**value,'citations':[{'evidence_id':'a','quote':'fabricated source text'}]},
                    {**value,'citations':[]},{**value,'status':'unsupported-status'}]:
            with self.assertRaises(ValueError): validate_reader_output(json.dumps(bad),[row])

    def test_refusal_cannot_contain_an_answer_or_citations(self):
        refusal={'status':'not_enough_evidence','answer':'','citations':[]}
        self.assertFalse(validate_reader_output(json.dumps(refusal),[])['answer_generated'])
        with self.assertRaises(ValueError): validate_reader_output(json.dumps({**refusal,'answer':'unverified'}),[])


if __name__=='__main__': unittest.main()
