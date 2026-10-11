import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,MagicMock,patch

from evidence_lab.source_recovery import recovery_eligible,recover_empty_reader
from evidence_lab.visual_evidence import VisualEvidence
from evidence_lab.native_pdf import SourceVerifier
from evidence_lab.reader_context import native_window_projection,ReaderBudget
from evidence_lab.reader import bind_answer_source_ids, VisualOutputInvalid, prepare_reader_visual_inputs
from test_visual_evidence import fixture


class SourceRecoveryTests(unittest.TestCase):
    def context(self):
        return SimpleNamespace(citation_catalog={'image-one':{
            'role':'primary_retrieved_region','image':{'file':'crop.png'},
            'readable_viewport':{'status':'selected_native_pdf_window'}}})

    def empty(self):
        return {'answer_candidate_present':False,'raw_output_for_validation':'{"answer":"","evidence_ids":[]}',
                'structure_validation':{'status':'passed'},'source_binding_validation':{'status':'passed'}}

    def test_answered_invalid_and_full_page_inputs_cannot_trigger_recovery(self):
        self.assertTrue(recovery_eligible(self.empty(),self.context()))
        for key,value in (('answer_candidate_present',True),('structure_validation',{}),
                          ('source_binding_validation',{}),('raw_output_for_validation','bad JSON'),
                          ('raw_output_for_validation','{"answer":"","evidence_ids":["image-one"]}')):
            value_empty=self.empty();value_empty[key]=value
            self.assertFalse(recovery_eligible(value_empty,self.context()))
        context=self.context();context.citation_catalog['image-one']['image']['file']='page.png'
        self.assertFalse(recovery_eligible(self.empty(),context))
        context=self.context();context.citation_catalog['image-one']['readable_viewport']['full_page_recovery']=True
        context.citation_catalog['image-one']['readable_viewport']['status']='full_page_fallback'
        self.assertFalse(recovery_eligible(self.empty(),context))

    def test_ineligible_answer_has_no_files_no_reread_and_same_object(self):
        generated=self.empty();generated['answer_candidate_present']=True
        with tempfile.TemporaryDirectory() as root:
            output=Path(root)/'must-not-exist';context=self.context()
            candidate,returned,report=recover_empty_reader('Describe the record',generated,context,
                hits=[],selected=[],visual=None,reader=None,processor=None,budget=None,config={},output=output)
            self.assertIs(candidate,generated);self.assertIs(returned,context)
            self.assertFalse(output.exists());self.assertEqual(report['actual_reader_calls'],0)

    def invalid_empty_reference(self):
        return {'answer_candidate_present':False,
            'raw_output_for_validation':'{"answer":"","evidence_ids":["image-one"]}',
            'actual_reader_run':True,'vision_inference_performed':True,
            'generation_provenance':'fixed_local_weights_native_visual_batch',
            'output_validation_error':{'stage':'structure',
                'reason':str(VisualOutputInvalid('structure',
                    'Empty answer must have no evidence; nonempty answer needs evidence'))}}

    def test_invalid_empty_known_reference_requests_new_call_only_when_enabled(self):
        value=self.invalid_empty_reference();original=copy.deepcopy(value)
        context=self.context()
        self.assertFalse(recovery_eligible(value,context))
        self.assertTrue(recovery_eligible(value,context,allow_empty_known_references=True))
        self.assertEqual(value,original)
        self.assertFalse(value['answer_candidate_present'])
        self.assertNotIn('structure_validation',value)

    def test_unknown_malformed_or_nonempty_failure_does_not_trigger_reread(self):
        base=self.invalid_empty_reference()
        bad_outputs=[
            '{"answer":"","evidence_ids":["unknown"]}',
            '{"answer":"","evidence_ids":["image-one"],"answer":""}',
            '{"answer":"","evidence_ids":[],"evidence_ids":["image-one"]}',
            'prefix {"answer":"","evidence_ids":["image-one"]}',
            '{"answer":" ","evidence_ids":["image-one"]}',
            '{"answer":"unsupported","evidence_ids":["image-one"]}',
            '{"answer":"","evidence_ids":[null]}',
            '{"answer":"","evidence_ids":[],"supported":true}',
            json.dumps({'answer':'','evidence_ids':['image-one']*9}),
        ]
        for raw in bad_outputs:
            value=dict(base,raw_output_for_validation=raw)
            with self.subTest(raw=raw):
                self.assertFalse(recovery_eligible(value,self.context(),allow_empty_known_references=True))

    def test_invalid_empty_requires_actual_visual_generation_and_expected_failure(self):
        for key,value in (('actual_reader_run',False),('vision_inference_performed',False),
                          ('generation_provenance','unverified'),
                          ('output_validation_error',{'stage':'source_binding','reason':'Unknown source image ID'}),
                          ('output_validation_error',{}),('answer_candidate_present',True)):
            candidate=dict(self.invalid_empty_reference(),**{key:value})
            with self.subTest(key=key):
                self.assertFalse(recovery_eligible(candidate,self.context(),allow_empty_known_references=True))
        context=self.context();context.citation_catalog['image-one']['image']['file']='page.png'
        self.assertFalse(recovery_eligible(self.invalid_empty_reference(),context,allow_empty_known_references=True))

    def test_actual_registered_pdf_binding_failure_requests_recovery_but_stays_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            _,_,browser,rows=fixture(root)
            visual=VisualEvidence(SourceVerifier(browser,rows))
            eid=rows[0]['evidence_id'];target=Path(root)/'source'
            visual.prepare_region(eid,target,
                readable_question='What responsibility does the Clerk have?')
            prepared=prepare_reader_visual_inputs('What responsibility does the Clerk have?',
                [rows[0]],visual=visual,primary_artifacts={eid:target},
                prompt_projection='answer_source_ids_v3')
            source_id=next(iter(prepared.citation_catalog))
            raw=json.dumps({'answer':'','evidence_ids':[source_id]})
            with self.assertRaises(VisualOutputInvalid) as caught:
                bind_answer_source_ids(raw,prepared)
            exc=caught.exception
            self.assertEqual(exc.stage,'structure')
            self.assertTrue(str(exc).startswith('structure:'))
            generated=self.invalid_empty_reference()
            generated['raw_output_for_validation']=raw
            generated['output_validation_error']={'stage':exc.stage,'reason':str(exc)}
            before=copy.deepcopy(generated)
            self.assertTrue(recovery_eligible(generated,prepared,allow_empty_known_references=True))
            self.assertEqual(generated,before)
            self.assertFalse(generated['answer_candidate_present'])
            with self.assertRaises(VisualOutputInvalid):
                bind_answer_source_ids(raw,prepared)
            # The same empty-answer fault still precedes the unknown-ID test
            # inside the binder, so acquisition must independently reject IDs.
            generated['raw_output_for_validation']=json.dumps({'answer':'','evidence_ids':['unknown']})
            self.assertFalse(recovery_eligible(generated,prepared,allow_empty_known_references=True))

    def test_full_original_page_reproduces_pixels_and_keeps_document_identity(self):
        with tempfile.TemporaryDirectory() as root:
            _,_,browser,rows=fixture(root);visual=VisualEvidence(SourceVerifier(browser,rows))
            eid=rows[0]['evidence_id'];original=visual.prepare_region(eid,Path(root)/'first',
                readable_question='What responsibility does the Clerk have?')
            recovery=visual.prepare_region(eid,Path(root)/'recovery',
                readable_question='What responsibility does the Clerk have?',full_page_recovery=True)
            self.assertEqual(original['source'],recovery['source'])
            self.assertNotEqual(original['selected_image']['rgb_pixels_sha256'],recovery['selected_image']['rgb_pixels_sha256'])
            self.assertEqual(recovery['selected_image']['rgb_pixels_sha256'],recovery['page_image']['rgb_pixels_sha256'])
            self.assertTrue(visual.verify(Path(root)/'recovery')['all_passed'])
            self.assertTrue(recovery['readable_viewport']['full_page_recovery'])
            self.assertFalse(recovery['supported_answer'])

    def test_whole_page_prose_does_not_become_a_truncated_native_table(self):
        original='Literal original region';viewport={'full_page_recovery':True,
            'native_spatial_word_lines':[['Caption','Name'],['10','20']]}
        before=copy.deepcopy(viewport)
        self.assertEqual(native_window_projection(viewport,original),original)
        self.assertEqual(viewport,before)

    def test_nonempty_answer_never_queries_supplementary_features(self):
        generated=self.empty();generated['answer_candidate_present']=True
        bank=Mock()
        with tempfile.TemporaryDirectory() as root:
            value,_,report=recover_empty_reader('Alpha Delta',generated,self.context(),
                hits=[],selected=[],visual=None,reader=None,processor=None,budget=None,
                config={'bounded_native_image_source_recovery':True},output=Path(root)/'output',feature_search=bank)
        self.assertIs(value,generated);bank.supplementary_source.assert_not_called()
        self.assertEqual(report['actual_reader_calls'],0)

    def test_full_page_empty_input_can_request_new_source_but_bad_proof_cannot_run_reader(self):
        context=self.context();context.citation_catalog['image-one']['image']['file']='page.png'
        bank=Mock();bank.supplementary_source.return_value={'canonical_source':{'evidence_id':'new-original'}}
        visual=Mock();visual.verifier.verify.return_value={'all_passed':False}
        reader=Mock();original=self.empty()
        with tempfile.TemporaryDirectory() as root:
            value,returned,report=recover_empty_reader('Alpha Delta',original,context,
                hits=[],selected=[],visual=visual,reader=reader,processor=None,budget=None,
                config={'bounded_native_image_source_recovery':True},output=Path(root)/'output',feature_search=bank)
            self.assertTrue((Path(root)/'output/report.json').exists())
        self.assertIs(value,original);self.assertIs(returned,context)
        self.assertEqual(report['status'],'recovery_failed_original_refusal_retained')
        self.assertEqual(report['actual_reader_calls'],0)
        reader.read_visual.assert_not_called();visual.prepare_region.assert_not_called()

    def test_actual_new_registered_source_without_full_page_audit_is_budget_checked_once(self):
        for valid_budget,full_page in ((True,False),(False,False),(True,True),(False,True)):
            window=(patch('evidence_lab.readable_region.choose_window',side_effect=lambda question,*args:{
                'status':'full_page_fallback','reason':'no_bounded_native_table_or_image_window',
                'question':question,'association_verified':False}) if full_page else nullcontext())
            with self.subTest(valid_budget=valid_budget,full_page=full_page),tempfile.TemporaryDirectory() as root,window:
                _,doc,browser,rows=fixture(root);visual=VisualEvidence(SourceVerifier(browser,rows))
                source=rows[0];eid=source['evidence_id']
                context=SimpleNamespace(citation_catalog={'original-image':{
                    'role':'primary_retrieved_region','parent_evidence_id':'original-other-page',
                    'doc_sha256':doc['sha256'],'page':2,'image':{
                        'file':'page.png','rgb_pixels_sha256':'different-original-page-pixels'}}})
                bank=Mock();bank.supplementary_source.return_value={'canonical_source':source}
                tensor=MagicMock();tensor.__getitem__.return_value.tolist.return_value=[[1]]
                grid=Mock();grid.tolist.return_value=[[1,10,10]]
                native=SimpleNamespace(tensors={'input_ids':tensor,'image_grid_thw':grid},
                    report={'budget_validation':{'token_budget_verified':valid_budget,'tensor_contract_validated':True}})
                reader=SimpleNamespace(context_tokenizer=None,read_visual=Mock(return_value=self.empty()))
                config={'bounded_native_image_source_recovery':True,'render_dpi':100,'max_new_tokens':512,
                    'prompt_projection':'answer_source_ids_v3','generation_seconds':10}
                original=self.empty()
                with patch('evidence_lab.reader_processor.prepare_native_visual_inputs',return_value=native):
                    candidate,prepared,report=recover_empty_reader('What responsibility does the Clerk have?',original,context,
                        hits=[],selected=[],visual=visual,reader=reader,processor=None,budget=ReaderBudget(),
                        config=config,output=Path(root)/'recovery',feature_search=bank)
                self.assertEqual(report['actual_reader_calls'],int(valid_budget),report)
                self.assertEqual(reader.read_visual.call_count,int(valid_budget))
                self.assertTrue(report['source_changes'][0]['actual_pixels_changed'])
                self.assertIsNone(report['source_changes'][0]['native_paint_filter'])
                self.assertFalse(report['source_changes'][0]['native_paint_filter_applied'])
                if full_page:
                    self.assertIsNone(report['source_changes'][0]['recovery_viewport'])
                self.assertEqual(report['source_changes'][0]['recovery_geometry']['page_rotation_clockwise'],0)
                self.assertEqual(report['source_changes'][0]['recovery_image_bbox_pixels'][:2],[0,0])
                self.assertFalse(report['semantic_support_established'])
                self.assertFalse(candidate['answer_candidate_present'])
                if valid_budget:
                    self.assertEqual(report['status'],'recovery_reader_completed')
                    self.assertEqual(next(iter(prepared.citation_catalog.values()))['parent_evidence_id'],eid)
                else:
                    self.assertIs(candidate,original);self.assertIs(prepared,context)
                    self.assertEqual(report['status'],'blocked_native_budget_no_reader_call')
