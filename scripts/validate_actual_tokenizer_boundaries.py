"""Actual pinned tokenizer budget tests; no processor/model construction, generation or weights."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from unittest.mock import patch

from evidence_lab.reader import LocalQwenReader
from evidence_lab.reader_context import (PinnedReaderTokenizer, PromptEncoding, ReaderBudget,
    assemble_reader_context, clean_reader_rows, reader_messages, source_binding)
from evidence_lab.safe_data import sha256_file


def source(eid: str, text: str, page: int = 1) -> dict:
    return {'role': 'source_pdf_region', 'evidence_id': eid, 'text': text,
            'document_name': 'self_authored_capacity_fixture', 'page': page, 'page_idx': page - 1,
            'doc_sha256': hashlib.sha256(b'self-authored tokenizer capacity fixture').hexdigest(),
            'limits': ['Self-authored capacity fixture, no actual PDF or answer-quality claim']}


def summary(context) -> dict:
    return {key: context.report[key] for key in ('status', 'ready_for_reader', 'token_budget_verified',
        'prompt_tokens', 'template_and_question_tokens', 'max_new_tokens_reserved', 'characters',
        'regions', 'retained_ids', 'omitted', 'text_truncated')}


def invariant(context, rows: list[dict], budget: ReaderBudget, reserved: int = 256) -> None:
    ids = [row['evidence_id'] for row in rows]
    kept = context.report['retained_ids']
    omitted = [item['evidence_id'] for item in context.report['omitted']]
    assert set(kept).isdisjoint(omitted) and set(kept) | set(omitted) == set(ids)
    assert kept == [eid for eid in ids if eid in kept]
    assert not context.report['text_truncated']
    if context.report['ready_for_reader']:
        clean = clean_reader_rows(rows)
        assert context.sources == [row for row in clean if row['evidence_id'] in kept]
        assert sum(len(row['text']) for row in context.sources) <= budget.max_characters
        assert len(context.sources) <= budget.max_regions
        assert len(context.encoding.token_ids) <= budget.max_prompt_tokens
        assert len(context.encoding.token_ids) + reserved <= budget.max_total_tokens
        assert [source_binding(row, i) for i, row in enumerate(clean)] == context.report['input_source_bindings']
    else:
        assert context.sources == [] and context.messages == [] and context.encoding is None


class IdentityCopyCounter(PinnedReaderTokenizer):
    """Independent-review counterexample: public metadata, no controlled loader."""
    def __init__(self, identity):
        self._synthetic_identity = copy.deepcopy(identity)
        self.calls = 0

    @property
    def identity(self):
        return self._synthetic_identity

    def encode_prompt(self, messages):
        self.calls += 1
        return PromptEncoding(tuple(range(101)), 'synthetic_no_template_hash')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--coverage-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.coverage_run.resolve()):
        raise ValueError('Use a new output directory outside preserved artifacts')
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    from transformers import AutoProcessor, PreTrainedModel, ProcessorMixin
    from transformers.generation.utils import GenerationMixin
    scenarios = []
    budget = ReaderBudget()
    with (patch.object(socket.socket, 'connect', side_effect=RuntimeError('Offline tokenizer regression')) as network,
          patch.object(AutoProcessor, 'from_pretrained', side_effect=RuntimeError('No processor allowed')) as processors,
          patch.object(ProcessorMixin, 'apply_chat_template', side_effect=RuntimeError('No processor call allowed')) as processor_calls,
          patch.object(PreTrainedModel, 'from_pretrained', side_effect=RuntimeError('No model allowed')) as models,
          patch.object(GenerationMixin, 'generate', side_effect=RuntimeError('No generation allowed')) as generation,
          patch.object(LocalQwenReader, '__init__', side_effect=RuntimeError('No Reader construction allowed')) as readers):
        tokenizer = PinnedReaderTokenizer(args.tokenizer_dir)
        protocol = json.loads(Path('configs/reader-context-v1.regression.json').read_text(encoding='utf-8'))
        for name, digest in protocol['inputs'].items():
            assert sha256_file(args.coverage_run / name) == digest
            data = json.loads((args.coverage_run / name).read_text(encoding='utf-8'))
            rows = data['fixed_payload']['hits'] + data['fixed_payload']['context']['added_regions']
            question = data['case']['question']
            original = copy.deepcopy(rows)
            context = assemble_reader_context(question, rows, tokenizer=tokenizer)
            invariant(context, rows, budget)
            assert context.report['ready_for_reader'] and context.report['prompt_tokens'] == 3466
            assert context.report['regions'] == 4 and context.report['characters'] == 11627
            assert tokenizer.encode_prompt(context.messages) == context.encoding
            assert assemble_reader_context(question, rows, tokenizer=tokenizer).encoding == context.encoding
            full_tokens = len(tokenizer.encode_prompt(reader_messages(question, clean_reader_rows(rows))).token_ids)
            scenarios.append({'case': name, 'kind': 'frozen_real_source_input', 'input_sha256': digest,
                              'original_full_five_region_prompt_tokens': full_tokens, **summary(context)})
            n = context.report['prompt_tokens']
            exact = ReaderBudget(max_prompt_tokens=n)
            below = ReaderBudget(max_prompt_tokens=n - 1)
            for label, limits in [('prompt_exact_boundary', exact), ('prompt_one_token_below', below),
                                  ('reserve_exact_boundary', ReaderBudget(max_total_tokens=n + 256)),
                                  ('reserve_one_token_below', ReaderBudget(max_total_tokens=n + 255))]:
                trial = assemble_reader_context(question, rows, tokenizer=tokenizer, budget=limits)
                invariant(trial, rows, limits)
                if 'exact' in label:
                    assert trial.report['retained_ids'] == context.report['retained_ids']
                else:
                    assert trial.report['retained_ids'] != context.report['retained_ids']
                    reason = 'prompt_token_budget' if label.startswith('prompt') else 'total_token_budget'
                    assert any(reason in item['reasons'] for item in trial.report['omitted'])
                scenarios.append({'case': name + '/' + label, 'kind': 'actual_tokenizer_real_source_boundary', **summary(trial)})
            assert rows == original

        question = 'Self-authored tokenizer capacity diagnostic'
        tiny = [source('tiny', '完整原文')]
        overhead = len(tokenizer.encode_prompt(reader_messages(question, [])).token_ids)
        low = ReaderBudget(max_prompt_tokens=overhead - 1)
        no_room = assemble_reader_context(question, tiny, tokenizer=tokenizer, budget=low)
        invariant(no_room, tiny, low)
        assert no_room.report['status'] == 'prompt_overhead_exceeds_budget'
        scenarios.append({'case': 'template_overhead_exceeds_budget', 'kind': 'self_authored_actual_tokenizer', **summary(no_room)})
        reader = LocalQwenReader.__new__(LocalQwenReader)
        reader.context_tokenizer = tokenizer
        refusal = reader.read(question, tiny, context_budget=low)
        assert refusal['status'] == 'not_enough_evidence' and not refusal['actual_reader_run']
        assert not refusal['answer_generated'] and refusal['answer'] == '' and refusal['citations'] == []

        for label, rows, limits, expected in (
            ('character_exact_boundary', [source('characters', 'x' * 12000)], budget, ['characters']),
            ('character_overflow_skip_whole', [source('too_long', 'x' * 12001), *tiny], budget, ['tiny']),
            ('region_exact_boundary', [source('region_' + str(i), '源', i + 1) for i in range(14)], budget,
             ['region_' + str(i) for i in range(13)])):
            context = assemble_reader_context(question, rows, tokenizer=tokenizer, budget=limits)
            invariant(context, rows, limits)
            assert context.report['retained_ids'] == expected
            scenarios.append({'case': label, 'kind': 'self_authored_actual_tokenizer', **summary(context)})

        long_tokens = source('too_many_tokens', '汉' * 5000)
        measured = len(tokenizer.encode_prompt(reader_messages(question, [long_tokens])).token_ids)
        assert measured > 4096 and len(long_tokens['text']) < 12000
        context = assemble_reader_context(question, [long_tokens, *tiny], tokenizer=tokenizer)
        invariant(context, [long_tokens, *tiny], budget)
        assert context.report['retained_ids'] == ['tiny']
        assert context.report['omitted'][0]['reasons'] == ['prompt_token_budget', 'total_token_budget']
        scenarios.append({'case': 'whole_region_token_overflow_skip', 'kind': 'self_authored_actual_tokenizer',
                          'oversize_region_full_prompt_tokens': measured, **summary(context)})

        # Calibrate a self-authored fixture using actual tokenization to exercise the hard4096/4608 limits.
        seed_tokens = len(tokenizer.encode_prompt(reader_messages(question, [source('hard_prompt', '汉')])).token_ids)
        length = 4096 - seed_tokens + 1
        exact_source = source('hard_prompt', '汉' * length)
        assert len(tokenizer.encode_prompt(reader_messages(question, [exact_source])).token_ids) == 4096
        exact_context = assemble_reader_context(question, [exact_source], tokenizer=tokenizer, max_new_tokens=512)
        invariant(exact_context, [exact_source], budget, 512)
        assert exact_context.report['ready_for_reader'] and exact_context.report['prompt_tokens'] + 512 == 4608
        scenarios.append({'case': 'hard_4096_prompt_plus_512_reserve_exact', 'kind': 'self_authored_actual_tokenizer', **summary(exact_context)})
        overflow_source = source('hard_prompt', '汉' * (length + 1))
        assert len(tokenizer.encode_prompt(reader_messages(question, [overflow_source])).token_ids) == 4097
        overflow = assemble_reader_context(question, [overflow_source], tokenizer=tokenizer, max_new_tokens=512)
        invariant(overflow, [overflow_source], budget, 512)
        assert not overflow.report['ready_for_reader'] and overflow.report['omitted'][0]['trial_prompt_tokens'] == 4097
        scenarios.append({'case': 'hard_4097_prompt_rejected', 'kind': 'self_authored_actual_tokenizer', **summary(overflow)})
        reserve_low = ReaderBudget(max_total_tokens=4607)
        reserve_overflow = assemble_reader_context(question, [exact_source], tokenizer=tokenizer,
                                                    budget=reserve_low, max_new_tokens=512)
        invariant(reserve_overflow, [exact_source], reserve_low, 512)
        assert reserve_overflow.report['omitted'][0]['reasons'] == ['total_token_budget']
        scenarios.append({'case': 'hard_generation_reserve_overflow_rejected', 'kind': 'self_authored_actual_tokenizer', **summary(reserve_overflow)})
        refusal = reader.read(question, [exact_source], context_budget=reserve_low, max_new_tokens=512)
        assert not refusal['actual_reader_run'] and refusal['answer'] == '' and refusal['citations'] == []

        fake = IdentityCopyCounter(tokenizer.identity)
        forged = assemble_reader_context(question, tiny, tokenizer=fake)
        invariant(forged, tiny, budget)
        assert not forged.report['token_budget_verified'] and fake.calls == 0
        scenarios.append({'case': 'review_copied_public_identity_counter', 'kind': 'constructor_provenance_regression',
                          'synthetic_encoder_calls': fake.calls, **summary(forged)})
        bare = PinnedReaderTokenizer.__new__(PinnedReaderTokenizer)
        bare._tokenizer = tokenizer._tokenizer
        bare.__dict__['identity'] = copy.deepcopy(tokenizer.identity)
        copied = assemble_reader_context(question, tiny, tokenizer=bare)
        invariant(copied, tiny, budget)
        assert not copied.report['token_budget_verified']
        scenarios.append({'case': 'unregistered_exact_class_copied_backend', 'kind': 'constructor_provenance_regression', **summary(copied)})

        exported = tokenizer.identity
        exported['files']['tokenizer.json']['git_blob_sha1'] = 'synthetic_metadata_mutation'
        assert tokenizer.identity['files']['tokenizer.json']['git_blob_sha1'] != 'synthetic_metadata_mutation'
        assert assemble_reader_context(question, tiny, tokenizer=tokenizer).report['ready_for_reader']
        saved_template = tokenizer._tokenizer.chat_template
        try:
            tokenizer._tokenizer.chat_template = 'SYNTHETIC_REPLACEMENT_TEMPLATE'
            changed = assemble_reader_context(question, tiny, tokenizer=tokenizer)
            invariant(changed, tiny, budget)
            assert not changed.report['token_budget_verified']
            scenarios.append({'case': 'loaded_backend_template_changed', 'kind': 'live_loading_state_regression', **summary(changed)})
        finally:
            tokenizer._tokenizer.chat_template = saved_template
        assert assemble_reader_context(question, tiny, tokenizer=tokenizer).report['ready_for_reader']
        assert network.call_count == processors.call_count == processor_calls.call_count == models.call_count == generation.call_count == readers.call_count == 0
        result = {'kind': 'Actual pinned tokenizer-only full-prompt boundary regression; no answer-quality claim',
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'working_tree_clean': not subprocess.check_output(['git', 'status', '--porcelain']),
            'complete': True, 'scenario_count': len(scenarios), 'tokenizer_identity': tokenizer.identity,
            'budget': {'max_characters': 12000, 'max_regions': 13, 'max_prompt_tokens': 4096, 'max_total_tokens': 4608},
            'reserved_generation_tokens': 256, 'socket_connect_attempts': network.call_count,
            'processor_construction_calls': processors.call_count, 'processor_template_calls': processor_calls.call_count,
            'model_loading_calls': models.call_count, 'generation_calls': generation.call_count,
            'reader_construction_calls': readers.call_count, 'weights_loaded': False,
            'provided_qa_labels_opened': False, 'answer_accuracy_measured': False, 'scenarios': scenarios,
            'limits': ['Actual processor/tokenizer token equality remains untested because no processor was invoked',
                       'Synthetic source capacity fixtures are not actual PDF questions or semantic tests',
                       'Construction provenance prevents accidental substitutes, not hostile arbitrary in-process code']}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'actual-tokenizer-boundaries.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('scenarios', 'tokenizer_identity', 'limits')}, indent=2))


if __name__ == '__main__': main()
