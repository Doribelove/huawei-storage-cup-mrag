"""Native processor-only preflight; no downloads, weights, images or generation."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import traceback
from unittest.mock import patch

from evidence_lab.reader import LocalQwenReader
from evidence_lab.reader_context import (MODEL_ID, REVISION, TOKENIZER_FILES,
    PinnedReaderTokenizer, ReaderBudget, assemble_reader_context)
from evidence_lab.safe_data import sha256_file

PROCESSOR_FILES = {
    **TOKENIZER_FILES,
    'preprocessor_config.json': (390, '2ea84a437d448ff71b08df68fdd949d5cc4ebb64'),
    'video_preprocessor_config.json': (385, '3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe'),
}
PROTOCOL_SHA256 = '02e6faaa47d0ea913aff28032cb2868945a225664d139bce8398290a3f02dc6e'


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def snapshot(root: Path) -> dict:
    if not root.is_dir():
        raise ValueError('Preserved artifact directory is absent')
    return {str(p.relative_to(root)).replace('\\', '/'): sha256_file(p)
            for p in sorted(root.rglob('*')) if p.is_file()}


def verify_files(root: Path) -> dict:
    # A six-file allowlist also rejects tokenizer/template overrides and weights.
    if not root.is_dir() or {p.name for p in root.iterdir()} != set(PROCESSOR_FILES):
        raise ValueError('Processor-only directory must contain exactly the six pinned files')
    bindings = {}
    for name, (size, blob) in PROCESSOR_FILES.items():
        raw = (root / name).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if len(raw) != size or actual != blob:
            raise ValueError('Pinned processor file mismatch: ' + name)
        bindings[name] = {'bytes': size, 'git_blob_sha1': actual,
                          'sha256': hashlib.sha256(raw).hexdigest()}
    return bindings


def token_hash(ids) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def versions() -> dict:
    values = {}
    for name in ('transformers', 'torch', 'torchvision', 'tokenizers', 'pillow', 'numpy', 'protobuf'):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def processor_exit_code(report: dict) -> int:
    """0 means complete real text parity, 2 unavailable, 1 incomplete/mismatched."""
    if report.get('status') == 'processor_unavailable':
        return 2
    if (report.get('status') != 'text_processor_tokens_verified'
            or report.get('actual_text_processor_token_equality') is not True
            or report.get('investigation_complete') is not True):
        return 1
    construction = report.get('construction', {})
    if (construction.get('status') != 'constructed'
            or construction.get('class') != 'Qwen3VLProcessor'
            or report.get('calls', {}).get('native_processor_constructed') != 1):
        return 1
    cases = report.get('cases', [])
    if len(cases) != 2:
        return 1
    for case in cases:
        comparison = case.get('processor_text_comparison', {})
        if comparison.get('status') != 'compared' or any(comparison.get(key) is not True for key in (
                'token_ids_equal', 'rendered_sha256_equal', 'within_prompt_budget',
                'within_prompt_plus_generation_budget')):
            return 1
    return 0


def summarize_image_component(component, expected: dict) -> dict:
    """Report effective state, including loader-attribute/size drift; no images."""
    size = dict(component.size)
    minimum = getattr(component, 'min_pixels', None)
    maximum = getattr(component, 'max_pixels', None)
    # A native fast component can leave optional min/max attributes unset and
    # use size directly. A non-None attribute must agree with that size.
    consistent = ((minimum is None or minimum == size.get('shortest_edge'))
                  and (maximum is None or maximum == size.get('longest_edge')))
    matches = size == expected['size']
    shape_matches = all(getattr(component, name) == expected[name]
                        for name in ('patch_size', 'temporal_patch_size', 'merge_size'))
    return {'class': type(component).__name__, 'patch_size': component.patch_size,
            'temporal_patch_size': component.temporal_patch_size, 'merge_size': component.merge_size,
            'effective_size': size, 'min_pixels_attribute': minimum, 'max_pixels_attribute': maximum,
            'pinned_size': expected['size'], 'size_matches_pinned_config': matches,
            'pixel_attributes_agree_with_effective_size': consistent,
            'pinned_pixel_and_patch_settings_match': matches and consistent and shape_matches,
            'image_processing_performed': False, 'visual_token_budget_verified': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processor-dir', type=Path, required=True)
    parser.add_argument('--coverage-run', type=Path, required=True)
    parser.add_argument('--preserve-run', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # This known task worktree can be created by the sandbox account and read
    # by the explicitly approved host execution path. Trust only this path for
    # this command; never alter global safe.directory/ownership/security settings.
    git = ['git', '-c', 'safe.directory='+str(root.resolve())]
    preserved = [args.coverage_run, *args.preserve_run]
    if args.output.exists() or any(args.output.resolve().is_relative_to(p.resolve()) for p in preserved):
        raise ValueError('Use a new output directory outside preserved artifacts')
    bindings = verify_files(args.processor_dir)
    protocol_path = root / 'configs/reader-context-v1.regression.json'
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError('Frozen Reader protocol changed')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if sha256_file(args.coverage_run / 'agent-source-coverage-report.json') != protocol['coverage_report_sha256']:
        raise ValueError('Frozen coverage report changed')
    for name, digest in protocol['inputs'].items():
        if sha256_file(args.coverage_run / name) != digest:
            raise ValueError('Frozen Reader source changed: ' + name)
    before = {str(p.resolve()): snapshot(p) for p in preserved}
    args.output.mkdir(parents=True, exist_ok=False)
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'

    from transformers import AutoImageProcessor, AutoProcessor, PreTrainedModel, Qwen3VLProcessor
    from transformers.generation.utils import GenerationMixin
    processor = None
    cases = []
    calls = {'native_processor_construction_attempts': 0, 'native_processor_constructed': 0,
             'native_processor_render_calls': 0, 'native_processor_encode_calls': 0,
             'image_component_construction_attempts': 0, 'images_processed': 0,
             'tokenizer_context_assemblies': 0}
    # These guards protect prohibited actions; native processor construction and
    # its backend checks are unmodified. Nothing substitutes a dummy processor.
    with (patch.object(socket.socket, 'connect', side_effect=RuntimeError('Offline processor-only preflight')) as network,
          patch.object(PreTrainedModel, 'from_pretrained', side_effect=RuntimeError('No model loading authorized')) as models,
          patch.object(GenerationMixin, 'generate', side_effect=RuntimeError('No generation authorized')) as generation,
          patch.object(LocalQwenReader, '__init__', side_effect=RuntimeError('No Reader construction authorized')) as readers):
        calls['native_processor_construction_attempts'] += 1
        try:
            processor = AutoProcessor.from_pretrained(args.processor_dir,
                local_files_only=True, trust_remote_code=False)
        except Exception as error:
            construction = {'status': 'blocked', 'exception_type': type(error).__name__,
                            'message': str(error), 'traceback': traceback.format_exc()}
        else:
            if type(processor) is not Qwen3VLProcessor:
                raise TypeError('Native processor did not construct the exact Qwen3VLProcessor class')
            calls['native_processor_constructed'] += 1
            construction = {'status': 'constructed', 'class': type(processor).__name__,
                            'tokenizer_class': type(processor.tokenizer).__name__}

        # Component configuration is useful even if the required video backend
        # prevents full construction. This is not a full processor or an image run.
        calls['image_component_construction_attempts'] += 1
        try:
            component = AutoImageProcessor.from_pretrained(args.processor_dir,
                local_files_only=True, trust_remote_code=False)
            expected = json.loads((args.processor_dir / 'preprocessor_config.json').read_text(encoding='utf-8'))
            image_component = summarize_image_component(component, expected)
        except Exception as error:
            image_component = {'exception_type': type(error).__name__, 'message': str(error),
                               'image_processing_performed': False}

        tokenizer = PinnedReaderTokenizer(args.processor_dir)
        for name, digest in protocol['inputs'].items():
            original = json.loads((args.coverage_run / name).read_text(encoding='utf-8'))
            rows = original['fixed_payload']['hits'] + original['fixed_payload']['context']['added_regions']
            untouched = copy.deepcopy(rows)
            calls['tokenizer_context_assemblies'] += 1
            context = assemble_reader_context(original['case']['question'], rows, tokenizer=tokenizer,
                budget=ReaderBudget(**protocol['budget']), max_new_tokens=protocol['max_new_tokens'])
            assert context.report['ready_for_reader'] and context.encoding is not None
            assert context.report['prompt_tokens'] == 3466
            assert context.report['characters'] == protocol['expected_character_plan_characters']
            assert context.report['regions'] == protocol['expected_character_plan_regions']
            assert context.report['retained_ids'] == [r['evidence_id'] for r in rows[:4]]
            assert rows == untouched
            comparison = {'status': 'unverified_processor_unavailable', 'token_ids_equal': None,
                          'rendered_sha256_equal': None, 'actual_processor_prompt_tokens': None}
            if processor is not None:
                calls['native_processor_render_calls'] += 1
                rendered = processor.apply_chat_template(context.messages, add_generation_prompt=True, tokenize=False)
                calls['native_processor_encode_calls'] += 1
                from evidence_lab.reader_context import native_processor_messages
                native_messages = native_processor_messages(context.messages)
                calls['native_processor_render_calls'] += 1
                native_rendered = processor.apply_chat_template(native_messages, add_generation_prompt=True, tokenize=False)
                if native_rendered != rendered:
                    raise ValueError('Typed native content changed the whole original rendered prompt')
                inputs = processor.apply_chat_template(native_messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors='pt', truncation=False)
                assert inputs['input_ids'].ndim == 2 and inputs['input_ids'].shape[0] == 1
                assert not any(k in inputs for k in ('pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw'))
                ids = tuple(inputs['input_ids'][0].tolist())
                render_sha = hashlib.sha256(rendered.encode('utf-8')).hexdigest()
                comparison = {'status': 'compared', 'token_ids_equal': ids == context.encoding.token_ids,
                    'native_typed_content_render_equal': native_rendered == rendered,
                    'rendered_sha256_equal': render_sha == context.encoding.rendered_sha256,
                    'actual_processor_prompt_tokens': len(ids), 'actual_processor_token_ids_sha256': token_hash(ids),
                    'actual_processor_rendered_sha256': render_sha,
                    'within_prompt_budget': len(ids) <= context.report['budget']['max_prompt_tokens'],
                    'within_prompt_plus_generation_budget': len(ids) + protocol['max_new_tokens'] <= context.report['budget']['max_total_tokens']}
            item = {'input_file': name, 'input_sha256': digest, 'source_rows_and_ranking_unchanged': rows == untouched,
                    'actual_tokenizer_context': context.report,
                    'actual_tokenizer_token_ids_sha256': token_hash(context.encoding.token_ids),
                    'actual_tokenizer_rendered_sha256': context.encoding.rendered_sha256,
                    'processor_text_comparison': comparison, 'answer_generated': False, 'supported_answer': False}
            write(args.output / name.replace('.json', '-processor-check.json'), item)
            cases.append(item)
        calls.update(socket_connect_attempts=network.call_count, model_load_attempts=models.call_count,
                     generation_attempts=generation.call_count, reader_construction_attempts=readers.call_count)
        assert all(calls[k] == 0 for k in ('socket_connect_attempts', 'model_load_attempts',
                                         'generation_attempts', 'reader_construction_attempts'))
    after = {str(p.resolve()): snapshot(p) for p in preserved}
    assert before == after
    parity = None if processor is None else all(
        c['processor_text_comparison']['token_ids_equal'] and c['processor_text_comparison']['rendered_sha256_equal']
        and c['processor_text_comparison']['within_prompt_budget']
        and c['processor_text_comparison']['within_prompt_plus_generation_budget'] for c in cases)
    status = 'processor_unavailable' if processor is None else (
        'text_processor_tokens_verified' if parity else 'text_processor_tokens_mismatch')
    report = {'kind': 'Native Qwen processor-only preflight; no model/vision inference or accuracy measurement',
              'status': status, 'investigation_complete': True,
              'completed_utc': datetime.now(timezone.utc).isoformat(), 'model_id': MODEL_ID, 'revision': REVISION,
              'source_commit': subprocess.check_output([*git, 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
              'working_tree_clean': not subprocess.check_output([*git, 'status', '--porcelain'], cwd=root),
              'versions': versions(), 'protocol_sha256': PROTOCOL_SHA256, 'processor_files': bindings,
              'construction': construction, 'image_component_configuration': image_component,
              'actual_text_processor_token_equality': parity, 'calls': calls,
              'old_artifacts_preserved': before == after, 'old_artifact_sha256': before,
              'cases': [{'input_file': c['input_file'], 'input_sha256': c['input_sha256'],
                  'tokenizer_prompt_tokens': c['actual_tokenizer_context']['prompt_tokens'],
                  'generation_reserve': c['actual_tokenizer_context']['max_new_tokens_reserved'],
                  'processor_text_comparison': c['processor_text_comparison']} for c in cases],
              'source_file_sha256': {str(p.relative_to(root)).replace('\\', '/'): sha256_file(p) for p in (
                  Path(__file__), root / 'src/evidence_lab/reader.py', root / 'src/evidence_lab/reader_context.py')},
              'provided_qa_labels_opened': False, 'weights_present': False, 'dependency_installations': 0,
              'paid_api_cost': 0, 'answer_generated': False, 'supported_answer': False,
              'limits': ['Text-only tokens do not establish multimodal/image-token budgets',
                         'No processor substitute or backend bypass was used',
                         'Successful processor-only comparison would not establish Reader inference or semantic accuracy']}
    report['readiness_exit_code'] = processor_exit_code(report)
    report['exit_code_contract'] = {'0': 'Actual complete two-case text processor parity verified',
                                   '1': 'Incomplete, invalid or mismatched text parity',
                                   '2': 'Native processor unavailable; equality remains unverified'}
    write(args.output / 'reader-processor-report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('old_artifact_sha256', 'source_file_sha256', 'processor_files')}, ensure_ascii=False, indent=2))
    raise SystemExit(report['readiness_exit_code'])


if __name__ == '__main__':
    main()
