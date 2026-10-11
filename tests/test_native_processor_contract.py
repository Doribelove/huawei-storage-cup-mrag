"""Synthetic tensors/test-double mechanics; no native processor or model-quality proof."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import torch

from evidence_lab.reader import LocalQwenReader, ReaderUnavailable, _generate_visual_batch
from evidence_lab.reader_context import ReaderBudget, replay_reviewed_visual_context
from evidence_lab.reader_processor import (_validate_native_batch, NativeVisualInputs, PinnedVisualProcessor,
    VisualPixelPolicy, VisualProcessorUnavailable, native_visual_inputs_verified, prepare_native_visual_inputs)


IMAGE_TOKEN = 151655


def synthetic_batch():
    return {'input_ids':torch.tensor([[11]+[IMAGE_TOKEN]*64+[12]],dtype=torch.int64),
        'attention_mask':torch.ones((1,66),dtype=torch.int64),
        'image_grid_thw':torch.tensor([[1,16,16]],dtype=torch.int64),
        'pixel_values':torch.zeros((256,1536),dtype=torch.float32)}


def validate(batch, budget=ReaderBudget()):
    return _validate_native_batch(batch,(11,IMAGE_TOKEN,12),image_token_id=IMAGE_TOKEN,
        image_count=1,policy=VisualPixelPolicy(),budget=budget,max_new_tokens=256)


class ModelDouble:
    """An explicitly synthetic generate function, never a native model."""
    def __init__(self, mode='ok'):
        self.calls=0; self.mode=mode; self.kwargs=None; self.deadline_stopped=None
    def generate(self, **kwargs):
        self.calls += 1; self.kwargs=kwargs
        if self.mode=='deadline':
            self.deadline_stopped=bool(kwargs['stopping_criteria'](kwargs['input_ids'],None).item())
        prefix=kwargs['input_ids'].clone()
        if self.mode=='bad_prefix': prefix[0,0] += 1
        count=33 if self.mode=='too_long' else 3
        tail=torch.full((1,count),5,dtype=torch.int64)
        if self.mode=='negative': tail[0,0]=-1
        if self.mode=='empty': tail=tail[:,:0]
        return torch.cat([prefix,tail],dim=1)


class NativeProcessorContractTests(unittest.TestCase):
    def test_text_only_support_batch_needs_explicit_route(self):
        batch={'input_ids':torch.tensor([[11,12]],dtype=torch.int64),
               'attention_mask':torch.ones((1,2),dtype=torch.int64)}
        processor=Mock();processor.decode.return_value='synthetic support output'
        model=ModelDouble()
        with self.assertRaises(ReaderUnavailable):
            _generate_visual_batch(model,processor,batch,generation_reserve=32,max_new_tokens=32)
        self.assertEqual(model.calls,0)
        generated=_generate_visual_batch(model,processor,batch,generation_reserve=32,
            max_new_tokens=32,require_visual=False)
        self.assertEqual(generated['input_token_ids'],[11,12])
        self.assertEqual(generated['generated_token_ids'],[5,5,5])
        self.assertFalse(generated['supported_answer'])

    def test_text_support_route_rejects_incomplete_visual_pair(self):
        batch={'input_ids':torch.tensor([[11,12]],dtype=torch.int64),
               'attention_mask':torch.ones((1,2),dtype=torch.int64),'pixel_values':torch.zeros(1)}
        with self.assertRaises(ReaderUnavailable):
            _generate_visual_batch(ModelDouble(),Mock(),batch,generation_reserve=32,
                max_new_tokens=32,require_visual=False)

    def test_synthetic_grid_expansion_validates_mechanics_without_native_attestation(self):
        report=validate(synthetic_batch())
        self.assertEqual(report['expanded_prompt_tokens'],66)
        self.assertEqual(report['image_tokens_by_image'],[64])
        self.assertTrue(report['full_expanded_token_ids_equal'])
        self.assertNotIn('actual_native_processor_run',report)

    def test_actual_length_exceeds_cap_without_truncating_or_using_placeholder_count(self):
        report=validate(synthetic_batch(),ReaderBudget(max_prompt_tokens=65))
        self.assertEqual(report['placeholder_prompt_tokens'],3)
        self.assertEqual(report['expanded_prompt_tokens'],66)
        self.assertFalse(report['token_budget_verified'])

    def test_grid_count_temporal_merge_and_policy_mismatches_fail(self):
        for grid in (torch.tensor([[2,16,16]]),torch.tensor([[1,15,16]]),
                     torch.tensor([[1,2,2]]),torch.tensor([[1,16,16],[1,16,16]])):
            batch=synthetic_batch(); batch['image_grid_thw']=grid
            with self.assertRaises(VisualProcessorUnavailable): validate(batch)

    def test_bad_pixel_shape_dtype_and_nan_fail(self):
        for pixel in (torch.zeros((255,1536)),torch.zeros((256,1536),dtype=torch.float64)):
            batch=synthetic_batch(); batch['pixel_values']=pixel
            with self.assertRaises(VisualProcessorUnavailable): validate(batch)
        batch=synthetic_batch(); batch['pixel_values'][0,0]=float('nan')
        with self.assertRaises(VisualProcessorUnavailable): validate(batch)

    def test_prompt_prefix_image_count_mask_and_unknown_fields_fail(self):
        variants=[]
        batch=synthetic_batch(); batch['input_ids'][0,0]=99; variants.append(batch)
        batch=synthetic_batch(); batch['input_ids'][0,1]=99; variants.append(batch)
        batch=synthetic_batch(); batch['attention_mask'][0,0]=0; variants.append(batch)
        batch=synthetic_batch(); batch['pixel_values_videos']=torch.zeros(1); variants.append(batch)
        for batch in variants:
            with self.assertRaises(VisualProcessorUnavailable): validate(batch)

    def test_metadata_copies_do_not_register_source_processor_or_inputs(self):
        processor=PinnedVisualProcessor.__new__(PinnedVisualProcessor); processor._backend=Mock()
        prepared=Mock()
        with self.assertRaises(VisualProcessorUnavailable): prepare_native_visual_inputs(prepared,processor)
        inputs=NativeVisualInputs(prepared,processor,synthetic_batch(),
            {'budget_validation':{'token_budget_verified':True},'generation_authorized':True})
        self.assertFalse(native_visual_inputs_verified(inputs))
        reader=LocalQwenReader.__new__(LocalQwenReader); reader.model=Mock(); reader.processor=Mock()
        with self.assertRaises(ReaderUnavailable): reader.read_visual(inputs)
        reader.model.generate.assert_not_called(); reader.processor.apply_chat_template.assert_not_called()

    def test_generic_json_cannot_become_reviewed_source_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            handoff=Path(folder)/'fake.json'; handoff.write_text(json.dumps({'ready':True}),encoding='utf-8')
            with self.assertRaises(ValueError): replay_reviewed_visual_context(handoff,'native-commerce-page4')

    def test_adapter_double_receives_image_tensors_and_only_decodes_generated_tail(self):
        model=ModelDouble(); processor=Mock(); processor.decode.return_value='diagnostic output'
        batch=synthetic_batch(); result=_generate_visual_batch(model,processor,batch,max_new_tokens=32)
        self.assertEqual(model.calls,1)
        self.assertTrue(torch.equal(model.kwargs['pixel_values'],batch['pixel_values']))
        self.assertTrue(torch.equal(model.kwargs['image_grid_thw'],batch['image_grid_thw']))
        self.assertFalse(model.kwargs['do_sample'])
        self.assertEqual(processor.decode.call_args.args[0].tolist(),[5,5,5])
        self.assertEqual((result['input_tokens'],result['output_tokens']),(66,3))
        self.assertFalse(result['supported_answer'])

    def test_adapter_bad_requests_and_expanded_budget_fail_before_double_generate(self):
        for kwargs in ({'max_new_tokens':257},{'max_new_tokens':True},{'max_seconds':0},
                       {'budget':ReaderBudget(max_prompt_tokens=65)},
                       {'budget':ReaderBudget(max_total_tokens=321)}):
            model=ModelDouble()
            with self.assertRaises((ValueError,ReaderUnavailable)):
                _generate_visual_batch(model,Mock(),synthetic_batch(),**kwargs)
            self.assertEqual(model.calls,0)

    def test_adapter_corrupted_prefix_negative_or_over_budget_tail_is_withheld(self):
        for mode in ('bad_prefix','too_long','negative','empty'):
            model=ModelDouble(mode)
            with self.assertRaises(ReaderUnavailable):
                _generate_visual_batch(model,Mock(),synthetic_batch(),max_new_tokens=32)

    def test_adapter_per_token_deadline_and_late_completion_withhold_candidate(self):
        model=ModelDouble('deadline'); times=iter([0.,61.,61.])
        with self.assertRaises(ReaderUnavailable):
            _generate_visual_batch(model,Mock(),synthetic_batch(),max_seconds=60,_clock=lambda:next(times))
        self.assertTrue(model.deadline_stopped)

    def test_double_output_still_goes_through_unverified_citation_presentation_boundary(self):
        # Own PDF/source fixture is available in the unchanged full-test environment.
        from test_reader_visual_input import prepare_fixture
        from evidence_lab.reader import validate_visual_reader_output
        with tempfile.TemporaryDirectory() as folder:
            prepared,_,_,_=prepare_fixture(folder); image=next(iter(prepared.citation_catalog.values()))
            raw={'status':'answered','answer':'This is definitely the same table.',
                 'citations':[{'type':'text_quote','image_id':image['image_id'],'quote':'Body A'}]}
            decoder=Mock(); decoder.decode.return_value=json.dumps(raw)
            generated=_generate_visual_batch(ModelDouble(),decoder,synthetic_batch(),max_new_tokens=32)
            validated=validate_visual_reader_output(generated['raw_output_for_validation'],prepared)
            self.assertEqual((validated['status'],validated['answer'],validated['citations']),('not_enough_evidence','',[]))
            self.assertFalse(validated['supported_answer'])
            self.assertFalse(validated['evidence_support']['semantic_verifier_run'])


if __name__=='__main__': unittest.main()
