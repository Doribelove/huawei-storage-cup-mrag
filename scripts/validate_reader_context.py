"""Frozen source-only replay and all unit tests. Does not load a model or download anything."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from unittest.mock import patch

from evidence_lab.reader import LocalQwenReader, prepare_reader_sources, validate_reader_output
from evidence_lab.reader_context import (PinnedReaderTokenizer, ReaderBudget,
    assemble_reader_context, clean_reader_rows, source_binding)
from evidence_lab.safe_data import sha256_file


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)).replace('\\', '/'): sha256_file(p)
            for p in sorted(root.rglob('*')) if p.is_file()}


def rejected_quote(value: dict, rows: list[dict]) -> bool:
    try:
        validate_reader_output(json.dumps(value), rows)
    except ValueError:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coverage-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('configs/reader-context-v1.regression.json'))
    parser.add_argument('--tokenizer-dir', type=Path, help='Existing pinned local tokenizer files only; no download')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    if args.output.exists(): raise ValueError('Use a new output directory; never overwrite prior artifacts')
    if args.output.resolve().is_relative_to(args.coverage_run.resolve()):
        raise ValueError('New output must be outside the preserved coverage experiment directory')
    for name, digest in protocol['inputs'].items():
        if sha256_file(args.coverage_run / name) != digest: raise ValueError('Frozen source input hash mismatch: ' + name)
    if sha256_file(args.coverage_run / 'agent-source-coverage-report.json') != protocol['coverage_report_sha256']:
        raise ValueError('Frozen completed coverage report hash mismatch')
    before = snapshot(args.coverage_run)
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    with patch.object(socket.socket, 'connect', side_effect=RuntimeError('Reader boundary regression is offline')) as network:
        tokenizer = PinnedReaderTokenizer(args.tokenizer_dir) if args.tokenizer_dir else None
        for name in protocol['inputs']:
            original = json.loads((args.coverage_run / name).read_text(encoding='utf-8'))
            payload = original['fixed_payload']
            rows = payload['hits'] + payload['context']['added_regions']
            preserved = copy.deepcopy(rows)
            assert len(rows) == protocol['expected_original_regions']
            assert sum(len(row['text']) for row in rows) == protocol['expected_original_characters']
            try:
                prepare_reader_sources(rows)
            except ValueError as error:
                old_rejection = str(error)
            else:
                raise AssertionError('Frozen previously rejected input unexpectedly fits original character boundary')
            # Separate no-token character plan is diagnostic, with no prompt or generation permission.
            character_only = assemble_reader_context(original['case']['question'], rows,
                budget=ReaderBudget(**protocol['budget']), max_new_tokens=protocol['max_new_tokens'])
            plan = character_only.report['character_region_plan_only']
            assert plan['regions'] == protocol['expected_character_plan_regions']
            assert plan['characters'] == protocol['expected_character_plan_characters']
            assert plan['candidate_ids'] == [row['evidence_id'] for row in rows[:4]]
            omitted_id = protocol['expected_character_omitted_id']
            assert any(item['evidence_id'] == omitted_id and 'character_budget' in item['reasons']
                       for item in character_only.report['omitted'])
            plan_rows = prepare_reader_sources([row for row in rows if row['evidence_id'] in plan['candidate_ids']])
            expected = clean_reader_rows(rows)
            assert plan_rows == expected[:4]
            assert [source_binding(row, i) for i, row in enumerate(plan_rows)] == character_only.report['input_source_bindings'][:4]
            context = assemble_reader_context(original['case']['question'], rows, tokenizer=tokenizer,
                budget=ReaderBudget(**protocol['budget']), max_new_tokens=protocol['max_new_tokens'])
            if tokenizer is None:
                assert not context.report['ready_for_reader'] and not context.report['token_budget_verified']
                assert context.sources == [] and context.messages == [] and context.encoding is None
                reader = LocalQwenReader.__new__(LocalQwenReader)
                result = reader.read(original['case']['question'], rows)
                assert result['status'] == 'not_enough_evidence' and not result['actual_reader_run']
                assert not result['answer_generated'] and not result['supported_answer']
            else:
                # Tokenizer-only verification is allowed; no processor/model call follows a passing assembly.
                result = {'actual_reader_run': False, 'answer_generated': False, 'supported_answer': False,
                          'status': 'generation_not_attempted_in_tokenizer_only_regression'}
            # Self-authored output contract checks; these are not generated or semantically supported answers.
            candidate = {'status': 'answered', 'answer': 'Synthetic contract probe',
                         'citations': [{'evidence_id': omitted_id, 'quote': rows[-1]['text'][:32]}]}
            assert rejected_quote(candidate, plan_rows)
            candidate['citations'] = [{'evidence_id': rows[0]['evidence_id'], 'quote': 'SYNTHETIC_NON_ORIGINAL_QUOTE'}]
            assert rejected_quote(candidate, plan_rows)
            refusal = validate_reader_output(json.dumps({'status': 'not_enough_evidence', 'answer': '', 'citations': []}), plan_rows)
            assert not refusal['answer_generated'] and not refusal['supported_answer']
            assert rows == preserved
            item = {'input_file': name, 'input_sha256': protocol['inputs'][name],
                    'original_evidence_ids': [row['evidence_id'] for row in rows],
                    'original_character_rejection': old_rejection,
                    'character_plan_only': character_only.report,
                    'actual_tokenizer_assembly': context.report,
                    'source_rows_and_ranking_unchanged': rows == preserved,
                    'whole_original_char_plan_bindings_exact': True,
                    'synthetic_citation_contract_checks': {'omitted_id_rejected': True, 'changed_quote_rejected': True,
                                                          'empty_refusal_passed': True},
                    'boundary_result': result}
            write(args.output / name.replace('.json', '-reader-context.json'), item)
            reports.append(item)
        tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                               cwd=root, capture_output=True, text=True, encoding='utf-8', check=False)
        (args.output / 'unit-tests.txt').write_text(tests.stdout + tests.stderr, encoding='utf-8')
        if tests.returncode: raise RuntimeError('Unit tests failed; preserved output contains details')
        after = snapshot(args.coverage_run)
        assert before == after
        assert network.call_count == 0
    source_bindings = {str(path.relative_to(root)).replace('\\', '/'): sha256_file(path) for path in (
        root / 'src/evidence_lab/reader.py', root / 'src/evidence_lab/reader_context.py',
        root / 'scripts/validate_reader_context.py', root / 'tests/test_reader_context.py')}
    report = {'kind': 'No-model whole-region Reader boundary regression; not model availability or accuracy',
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
              'working_tree_clean': not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root),
              'complete': len(reports) == 2, 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': sha256_file(args.protocol), 'source_file_sha256': source_bindings,
              'source_base_commit': protocol['source_commit'],
              'actual_tokenizer_loaded': tokenizer is not None, 'actual_reader_inference': False,
              'actual_prompt_tokens_verified': tokenizer is not None and all(r['actual_tokenizer_assembly']['token_budget_verified'] for r in reports),
              'all_unit_tests_passed': tests.returncode == 0, 'socket_connect_attempts': network.call_count,
              'paid_api_cost': 0, 'provided_qa_labels_opened': False,
              'old_artifacts_preserved': before == after, 'old_artifact_count': len(before),
              'old_artifact_sha256': before,
              'cases': [{k: item[k] for k in ('input_file', 'input_sha256', 'source_rows_and_ranking_unchanged',
                  'whole_original_char_plan_bindings_exact', 'boundary_result')} for item in reports],
              'limits': protocol['limits']}
    write(args.output / 'reader-context-report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('old_artifact_sha256', 'source_file_sha256', 'cases')}, indent=2))


if __name__ == '__main__': main()
