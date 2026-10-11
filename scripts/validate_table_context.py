"""Two fixed source failures: native PDF header presence, not semantic accuracy."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import time
import traceback
from unittest.mock import patch

from evidence_lab.demo import EvidenceBrowser
from evidence_lab.native_pdf import SourceVerifier
from evidence_lab.safe_data import sha256_file
from evidence_lab.table_context import ContextBudget, TableContext, normalized
from evidence_lab.visual_evidence import VisualEvidence


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def probes(record, expectations):
    cells = record['extraction']['cells'][:3]
    columns = max(map(len, cells), default=0)
    result = []
    for label in expectations:
        locations = []
        for column in range(columns):
            source_cells = [{'row_idx': index, 'column_idx': column, 'source_text': row[column]}
                            for index, row in enumerate(cells) if column < len(row) and row[column]]
            text = ' '.join(normalized(cell['source_text']) for cell in source_cells)
            if ' '+normalized(label)+' ' in ' '+text+' ':
                locations.append({'page': record['page'], 'evidence_id': record['evidence_id'],
                                  'source_cells': source_cells, 'table_bbox': record['source_bbox']})
        result.append({'source_header_column': label, 'present': bool(locations), 'locations': locations})
    return result


def validate(args):
    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    previous = json.loads(args.previous_handoff.read_text(encoding='utf-8'))
    if previous['source_commit'] != protocol['base_commit']:
        raise ValueError('Use the accepted09ccc93 visual source handoff')
    previous_root = Path(previous['worktree'])
    def verify_previous():
        for member, expected in previous['artifact_sha256'].items():
            path = previous_root/member
            path.resolve().relative_to(previous_root.resolve())
            if sha256_file(path) != expected: raise ValueError('Accepted visual artifact changed: '+member)
    verify_previous()
    original_protocol = json.loads(Path('configs/visual-evidence-v1.json').read_text(encoding='utf-8'))
    if (sha256_file(args.source_regions) != original_protocol['full_source_regions_sha256']
            or sha256_file(args.audit/'regions.jsonl') != original_protocol['original_regions_sha256']):
        raise ValueError('Frozen source corpus changed')
    wanted = {case['evidence_id'] for case in protocol['cases']}
    overlays = []
    with args.source_regions.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['evidence_id'] in wanted: overlays.append(row)
    if len(overlays) != 2 or len(protocol['cases']) != 2:
        raise ValueError('This regression is limited to the two frozen source failures')
    browser = EvidenceBrowser(args.audit, args.phase, data_archive=args.data_archive,
                              cache_dir=args.output/'original-pdf-cache')
    if browser.audit['source_archives']['data']['sha256'] != original_protocol['data_archive_sha256']:
        raise ValueError('Frozen data archive changed')
    budget = ContextBudget(**{k: protocol[k] for k in ('max_backward_pages', 'max_unique_pages', 'max_image_assets', 'max_total_pixels')})
    visual = VisualEvidence(SourceVerifier(browser, overlays)); context = TableContext(visual, budget)
    results = []
    for case in protocol['cases']:
        primary = args.primary_root/case['case']; output = args.output/case['case']
        before = {p.name: sha256_file(p) for p in primary.iterdir() if p.is_file()}
        result = context.prepare(primary, output)
        if result['primary']['source']['region_evidence_id'] != case['evidence_id']:
            raise ValueError('Primary source ID differs from frozen failure case')
        verification = context.verify(output)
        before_probes = probes(visual.verifier.extra[case['evidence_id']], case['expected_header_columns'])
        after_probes = []
        for label in case['expected_header_columns']:
            locations = [loc for candidate in result['candidate_context'] if candidate['header_hints']
                         for loc in probes(candidate['source_record'], [label])[0]['locations']]
            after_probes.append({'source_header_column': label, 'present_in_candidate_page': bool(locations),
                                 'locations': locations, 'attached_to_primary_table_semantically': False})
        checks = {'source_context_and_image_replay': verification['all_passed'],
            'primary_files_unchanged': before == {p.name: sha256_file(p) for p in primary.iterdir() if p.is_file()},
            'primary_columns_missing_before': all(not probe['present'] for probe in before_probes),
            'all_missing_column_strings_present_in_candidate_after': all(probe['present_in_candidate_page'] for probe in after_probes),
            'expected_minimal_page_chain': [c['page'] for c in result['candidate_context']] == case['expected_candidate_pages'],
            'all_cross_page_relations_remain_candidates': all(not c['table_continuity_verified'] and not c['header_assigned_to_primary_table'] for c in result['candidate_context']),
            'no_answer_or_vision_claim': not result['supported_answer'] and not result['vision_reasoning_performed'],
            'aggregate_pixels_bounded': result['budget_usage']['total_selected_pixels'] <= budget.max_total_pixels}
        results.append({'case': case['case'], 'primary': result['primary'], 'candidate_pages': [c['page'] for c in result['candidate_context']],
            'same_page_source_line_added': result['same_page_context'], 'column_presence_before': before_probes,
            'column_presence_after_in_candidate': after_probes, 'budget_usage': result['budget_usage'],
            'stop_reason': result['stop_reason'], 'verification': verification, 'checks': checks, 'all_passed': all(checks.values())})
    verify_previous()
    return {'state': 'bounded_source_context_diagnostic_passed' if all(r['all_passed'] for r in results) else 'bounded_source_context_diagnostic_failed',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'working_tree_clean': not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(),
        'protocol_sha256': sha256_file(args.protocol), 'previous_handoff_sha256': sha256_file(args.previous_handoff),
        'accepted_visual_artifacts_before_after_hashes_unchanged': True, 'accepted_visual_artifact_count': len(previous['artifact_sha256']),
        'cases': results, 'actual_context_page_pngs': sum(len(r['candidate_pages']) for r in results),
        'actual_render_calls_including_source_replays': visual.render_calls,
        'elapsed_seconds': time.perf_counter()-started, 'qa_labels_opened': False,
        'retrieval_rankings_read_or_changed': False, 'table_continuity_verified': False,
        'visual_token_budget_verified': False, 'vision_reasoning_performed': False,
        'answer_accuracy_measured': False, 'supported_answer': False,
        'scope': 'Missing column strings are present in verified candidate source pages; their semantic association remains unverified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('configs/table-context-v1.json'))
    for name in ('audit', 'phase', 'data-archive', 'source-regions', 'primary-root', 'previous-handoff', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=False)
    network_attempts = 0
    def blocked(*_, **__):
        nonlocal network_attempts
        network_attempts += 1
        raise RuntimeError('Network disabled for this local source context diagnostic')
    try:
        with patch.object(socket.socket, 'connect', blocked), patch.object(socket, 'create_connection', blocked):
            report = validate(args)
        code = 0 if report['state'] == 'bounded_source_context_diagnostic_passed' else 1
    except Exception as exc:
        report = {'state': 'bounded_source_context_diagnostic_blocked', 'error_class': type(exc).__name__,
            'error': str(exc), 'traceback': traceback.format_exc(), 'vision_reasoning_performed': False, 'supported_answer': False}
        code = 2
    report['blocked_network_attempts'] = network_attempts
    write(args.output/'table-context-report.json', report)
    print(json.dumps({k: report[k] for k in ('state', 'vision_reasoning_performed', 'supported_answer')}, ensure_ascii=False))
    return code


if __name__ == '__main__': raise SystemExit(main())
