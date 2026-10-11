"""Whole-region and prompt-budget logic; synthetic token IDs are NOT Qwen token measurements."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from evidence_lab.reader import LocalQwenReader, ReaderUnavailable, validate_reader_output
from evidence_lab.reader_context import (PinnedReaderTokenizer, PromptEncoding, ReaderBudget,
    SYSTEM_PROMPT, MODEL_ID, REVISION, TOKENIZER_FILES, TokenizerUnavailable, _encode_full_prompt,
    _assemble_reader_context as assemble_reader_context, assemble_reader_context as production_assembly,
    reader_messages)
from test_pilot import fixture


class SyntheticTokenizer(PinnedReaderTokenizer):
    """Fixed per-ID token costs exercise budget arithmetic, never estimate actual tokens from chars."""
    def __init__(self, costs=None, overhead=10):
        self.costs = costs or {}
        self.overhead = overhead
        self.calls = []
        self._synthetic_identity = {'verification': 'synthetic_unit_test_only', 'weights_loaded': False}

    @property
    def identity(self):
        return self._synthetic_identity

    def encode_prompt(self, messages):
        self.calls.append(copy.deepcopy(messages))
        assert messages[0] == {'role': 'system', 'content': SYSTEM_PROMPT}
        payload = json.loads(messages[1]['content'])
        count = self.overhead + sum(self.costs.get(row['evidence_id'], 7) for row in payload['source_evidence'])
        return PromptEncoding(tuple(range(count)), hashlib.sha256(json.dumps(messages).encode()).hexdigest())


class ReaderContextTests(unittest.TestCase):
    def test_missing_actual_tokenizer_closes_gate_with_a_separate_character_plan(self):
        rows = [fixture('a', 0, text='A' * 8000), fixture('b', 1, text='B' * 5000), fixture('c', 2, text='完整区域')]
        before = copy.deepcopy(rows)
        context = assemble_reader_context('Question', rows)
        self.assertFalse(context.report['ready_for_reader'])
        self.assertFalse(context.report['token_budget_verified'])
        self.assertIsNone(context.report['prompt_tokens'])
        self.assertEqual((context.sources, context.messages, context.encoding), ([], [], None))
        self.assertEqual(context.report['character_region_plan_only']['candidate_ids'], ['a', 'c'])
        self.assertEqual([r['reasons'] for r in context.report['omitted']],
                         [['tokenizer_unavailable'], ['character_budget'], ['tokenizer_unavailable']])
        self.assertEqual(rows, before)
        with self.assertRaises(ValueError): assemble_reader_context('Question', rows, tokenizer=lambda _: 1)

    def test_character_budget_uses_whole_regions_and_continues_to_a_later_fit(self):
        rows = [fixture('long', 0, text='x' * 12001), fixture('a', 1, text='完整原文'), fixture('b', 2, text='尾部')]
        before = copy.deepcopy(rows)
        ctx = assemble_reader_context('Question', rows, tokenizer=SyntheticTokenizer(),
                                      budget=ReaderBudget(max_characters=4))
        self.assertEqual(ctx.sources, [rows[1]])
        self.assertEqual(ctx.report['retained_ids'], ['a'])
        self.assertEqual([r['evidence_id'] for r in ctx.report['omitted']], ['long', 'b'])
        self.assertTrue(all('character_budget' in r['reasons'] for r in ctx.report['omitted']))
        self.assertFalse(ctx.report['text_truncated'])
        self.assertEqual(rows, before)

    def test_region_limit_preserves_incoming_order_instead_of_resorting(self):
        rows = [fixture(str(i), i, text='完整') for i in range(14)]
        ctx = assemble_reader_context('Question', rows, tokenizer=SyntheticTokenizer())
        self.assertEqual(ctx.report['retained_ids'], [str(i) for i in range(13)])
        self.assertEqual(ctx.report['omitted'][0]['evidence_id'], '13')
        self.assertEqual(ctx.report['omitted'][0]['reasons'], ['region_budget'])

    def test_prompt_overhead_and_generation_prefix_are_counted_before_sources(self):
        tok = SyntheticTokenizer(overhead=20)
        ctx = assemble_reader_context('Question', [fixture('a', 0)], tokenizer=tok,
                                      budget=ReaderBudget(max_prompt_tokens=19))
        self.assertEqual(ctx.report['status'], 'prompt_overhead_exceeds_budget')
        self.assertFalse(ctx.report['ready_for_reader'])
        self.assertEqual(ctx.sources, [])
        self.assertEqual(ctx.report['template_and_question_tokens'], 20)
        self.assertEqual(ctx.report['omitted'][0]['reasons'], ['prompt_template_or_question_budget'])
        self.assertEqual(json.loads(tok.calls[0][1]['content'])['source_evidence'], [])

    def test_prompt_token_boundary_and_long_token_region_omission_without_char_proxy(self):
        # A one-character region costs100 fixture tokens; a long string costs5. These are deliberately independent.
        rows = [fixture('long_tokens', 0, text='短'), fixture('a', 1, text='x' * 500), fixture('b', 2, text='end')]
        tok = SyntheticTokenizer({'long_tokens': 100, 'a': 5, 'b': 1}, overhead=10)
        ctx = assemble_reader_context('Question', rows, tokenizer=tok,
                                      budget=ReaderBudget(max_prompt_tokens=15))
        self.assertEqual(ctx.report['retained_ids'], ['a'])
        self.assertEqual(ctx.report['prompt_tokens'], 15)
        self.assertEqual([r['reasons'] for r in ctx.report['omitted']],
                         [['prompt_token_budget'], ['prompt_token_budget']])
        self.assertEqual(ctx.sources[0]['text'], 'x' * 500)
        self.assertEqual(json.loads(ctx.messages[1]['content'])['source_evidence'][0]['text'], 'x' * 500)

    def test_generation_reserve_boundary_and_no_complete_fit_refusal(self):
        rows = [fixture('a', 0)]
        boundary = assemble_reader_context('Question', rows, tokenizer=SyntheticTokenizer({'a': 5}),
                   budget=ReaderBudget(max_total_tokens=47), max_new_tokens=32)
        self.assertEqual(boundary.report['prompt_tokens'] + 32, 47)
        no_fit = assemble_reader_context('Question', rows, tokenizer=SyntheticTokenizer({'a': 5}),
                   budget=ReaderBudget(max_total_tokens=46), max_new_tokens=32)
        self.assertEqual(no_fit.report['status'], 'no_complete_region_fits')
        self.assertFalse(no_fit.report['ready_for_reader'])
        self.assertEqual(no_fit.sources, [])
        self.assertEqual(no_fit.report['omitted'][0]['reasons'], ['total_token_budget'])
        empty = assemble_reader_context('Question', [fixture('empty', 0, text='  ')], tokenizer=SyntheticTokenizer())
        self.assertEqual(empty.report['omitted'][0]['reasons'], ['empty_source_text'])

    def test_all_source_associations_and_qa_fields_validated_even_if_region_would_be_omitted(self):
        row = fixture('a', 0)
        for bad in [dict(row, page=True), dict(row, page=0), dict(row, page_idx=3),
                    dict(row, doc_sha256=''), dict(row, document_name=''),
                    dict(row, text='x' * 12001, metadata={'gt_answer': 'label'})]:
            with self.assertRaises(ValueError): assemble_reader_context('Question', [bad])
        with self.assertRaises(ValueError): assemble_reader_context('Question', [row, row])
        with self.assertRaises(ValueError): assemble_reader_context('Question', [row] * 65)

    def test_lineage_hashes_and_omitted_citation_rejection(self):
        rows = [fixture('kept', 0, page=2, name='one', pdf='original_pdf', text='Original complete quote'),
                fixture('omitted', 1, page=3, name='two', pdf='other_pdf', text='Foreign complete quote')]
        ctx = assemble_reader_context('Question', rows, tokenizer=SyntheticTokenizer(),
                                      budget=ReaderBudget(max_regions=1))
        binding = ctx.report['input_source_bindings'][0]
        self.assertEqual((binding['doc_sha256'], binding['document_name'], binding['page']), ('original_pdf', 'one', 2))
        self.assertEqual(binding['text_sha256'], hashlib.sha256(rows[0]['text'].encode()).hexdigest())
        candidate = {'status': 'answered', 'answer': 'Candidate', 'citations': [{'evidence_id': 'kept', 'quote': 'Original complete quote'}]}
        result = validate_reader_output(json.dumps(candidate), ctx.sources)
        self.assertFalse(result['supported_answer'])
        for eid, quote in [('omitted', 'Foreign complete quote'), ('kept', 'Foreign complete quote')]:
            with self.assertRaises(ValueError):
                validate_reader_output(json.dumps(dict(candidate, citations=[{'evidence_id': eid, 'quote': quote}])), ctx.sources)

    def test_tokenizer_failure_discards_partial_acceptance(self):
        tok = SyntheticTokenizer()
        original = tok.encode_prompt
        def encode(messages):
            payload = json.loads(messages[1]['content'])
            if len(payload['source_evidence']) > 1: raise RuntimeError('diagnostic tokenizer failure')
            return original(messages)
        tok.encode_prompt = encode
        ctx = assemble_reader_context('Question', [fixture('a', 0), fixture('b', 1)], tokenizer=tok)
        self.assertEqual(ctx.report['status'], 'tokenizer_verification_failed')
        self.assertEqual((ctx.sources, ctx.messages, ctx.encoding), ([], [], None))
        self.assertEqual(ctx.report['characters'], 0)
        self.assertEqual([r['evidence_id'] for r in ctx.report['omitted']], ['a', 'b'])
        self.assertFalse(ctx.report['ready_for_reader'])

    def test_bad_budget_types_are_not_silently_coerced(self):
        rows = [fixture('a', 0)]
        for budget in [ReaderBudget(max_characters=True), ReaderBudget(max_regions=0),
                       ReaderBudget(max_prompt_tokens=4097), ReaderBudget(max_total_tokens=4609),
                       ReaderBudget(max_characters=1.5), {'max_regions': 1}]:
            with self.assertRaises(ValueError): assemble_reader_context('Question', rows, budget=budget)
        for question in ['', 'x' * 501, None]:
            with self.assertRaises(ValueError): assemble_reader_context(question, rows)

    def test_missing_or_modified_tokenizer_files_are_rejected_before_library_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(TokenizerUnavailable): PinnedReaderTokenizer(Path(folder))
            name, (size, _) = next(iter(TOKENIZER_FILES.items()))
            (Path(folder) / name).write_bytes(b' ' * size)
            with self.assertRaisesRegex(TokenizerUnavailable, 'hash mismatch'):
                PinnedReaderTokenizer(Path(folder))
            (Path(folder) / 'special_tokens_map.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(TokenizerUnavailable, 'Unverified.*sidecar'):
                PinnedReaderTokenizer(Path(folder))

    def test_pinned_tokenizer_counts_full_render_without_special_token_duplication_or_truncation(self):
        backend = Mock()
        backend.apply_chat_template.return_value = 'FULL_TEMPLATE_WITH_GENERATION_PREFIX'
        backend.return_value = {'input_ids': [9, 11, 13]}
        messages = reader_messages('Question', [fixture('a', 0)])
        encoded = _encode_full_prompt(backend, messages)
        self.assertEqual(encoded.token_ids, (9, 11, 13))
        backend.apply_chat_template.assert_called_once_with(messages, add_generation_prompt=True, tokenize=False)
        backend.assert_called_once_with('FULL_TEMPLATE_WITH_GENERATION_PREFIX', add_special_tokens=False,
                                        truncation=False, return_attention_mask=False)
        backend.return_value = {'input_ids': [True]}
        with self.assertRaises(TokenizerUnavailable): _encode_full_prompt(backend, messages)

    def test_processor_token_mismatch_stops_before_generation(self):
        import numpy as np
        reader = LocalQwenReader.__new__(LocalQwenReader)
        reader.context_tokenizer = SyntheticTokenizer()
        reader.processor = Mock()
        reader.model = Mock()
        context = assemble_reader_context('Question', [fixture('a', 0)], tokenizer=SyntheticTokenizer())
        # The new native API first validates rendered bytes, then full token IDs.
        reader.processor.apply_chat_template.side_effect = lambda *a,**kw: (
            {'input_ids':np.array([[999]])} if kw['tokenize'] else json.dumps(context.messages))
        with patch('evidence_lab.reader.assemble_reader_context', return_value=context):
            with self.assertRaises(ReaderUnavailable): reader.read('Question', [fixture('a', 0)])
        self.assertEqual(reader.processor.apply_chat_template.call_count,2)
        reader.model.generate.assert_not_called()

    def test_native_nontext_or_changed_render_refuses_before_tokenizing_and_generation(self):
        context = assemble_reader_context('Question',[fixture('a',0)],tokenizer=SyntheticTokenizer())
        for rendered in ({'input_ids':[999]},None,'Changed full template'):
            reader=LocalQwenReader.__new__(LocalQwenReader)
            reader.context_tokenizer=SyntheticTokenizer(); reader.processor=Mock(); reader.model=Mock()
            reader.processor.apply_chat_template.return_value=rendered
            with patch('evidence_lab.reader.assemble_reader_context',return_value=context):
                with self.assertRaises(ReaderUnavailable): reader.read('Question',[fixture('a',0)])
            self.assertEqual(reader.processor.apply_chat_template.call_count,1)
            reader.model.generate.assert_not_called()

    def test_unverified_or_other_model_counter_cannot_pass_the_production_boundary(self):
        tok = SyntheticTokenizer()
        ctx = production_assembly('Question', [fixture('a', 0)], tokenizer=tok)
        self.assertEqual(ctx.report['status'], 'tokenizer_identity_unverified')
        self.assertFalse(ctx.report['token_budget_verified'])
        self.assertFalse(ctx.report['ready_for_reader'])
        self.assertEqual(ctx.sources, [])
        self.assertEqual(tok.calls, [])
        reader = LocalQwenReader.__new__(LocalQwenReader)
        reader.context_tokenizer = tok
        reader.processor = Mock()
        reader.model = Mock()
        result = reader.read('Question', [fixture('a', 0)])
        self.assertFalse(result['actual_reader_run'])
        reader.processor.apply_chat_template.assert_not_called()
        reader.model.generate.assert_not_called()

    def test_public_identity_copy_cannot_attest_a_synthetic_subclass(self):
        tok = SyntheticTokenizer(overhead=101)
        tok._synthetic_identity = {'model_id': MODEL_ID, 'revision': REVISION,
            'verification': 'pinned_local_files', 'load_state_verified': True,
            'local_files_only': True, 'trust_remote_code': False,
            'files': {name: {'bytes': size, 'git_blob_sha1': digest} for name, (size, digest) in TOKENIZER_FILES.items()}}
        ctx = production_assembly('Question', [fixture('a', 0)], tokenizer=tok)
        self.assertEqual(ctx.report['status'], 'tokenizer_identity_unverified')
        self.assertFalse(ctx.report['ready_for_reader'])
        self.assertFalse(ctx.report['token_budget_verified'])
        self.assertIsNone(ctx.report['prompt_tokens'])
        self.assertEqual((ctx.sources, ctx.messages, ctx.encoding), ([], [], None))
        self.assertEqual(tok.calls, [])

    def test_exact_class_without_successful_loading_cannot_attest_itself(self):
        tok = PinnedReaderTokenizer.__new__(PinnedReaderTokenizer)
        tok._tokenizer = Mock()
        tok.__dict__['identity'] = {'model_id': MODEL_ID, 'revision': REVISION,
            'verification': 'pinned_local_files', 'load_state_verified': True}
        ctx = production_assembly('Question', [fixture('a', 0)], tokenizer=tok)
        self.assertFalse(ctx.report['ready_for_reader'])
        self.assertFalse(ctx.report['token_budget_verified'])
        with self.assertRaises(TokenizerUnavailable): tok.encode_prompt(reader_messages('Question', [fixture('a', 0)]))
        tok._tokenizer.apply_chat_template.assert_not_called()

    def test_missing_tokenizer_refusal_has_no_processor_or_model_invocation(self):
        reader = LocalQwenReader.__new__(LocalQwenReader)
        reader.processor = Mock()
        reader.model = Mock()
        result = reader.read('Question', [fixture('a', 0)])
        self.assertEqual((result['status'], result['answer'], result['citations']), ('not_enough_evidence', '', []))
        self.assertFalse(result['answer_generated'])
        self.assertFalse(result['supported_answer'])
        self.assertFalse(result['actual_reader_run'])
        reader.processor.apply_chat_template.assert_not_called()
        reader.model.generate.assert_not_called()


if __name__ == '__main__': unittest.main()
