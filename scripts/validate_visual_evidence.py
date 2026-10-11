"""Three fixed local source pages; no QA labels, dependency/model/network calls."""
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
from evidence_lab.native_pdf import KIND, SourceVerifier
from evidence_lab.safe_data import sha256_file
from evidence_lab.visual_evidence import VisualEvidence


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def validate(args):
    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    if len(protocol['cases']) != 3 or protocol['dpi'] != 100:
        raise ValueError('This diagnostic is limited to the fixed three source cases')
    if sha256_file(args.audit/'regions.jsonl') != protocol['original_regions_sha256']:
        raise ValueError('Original frozen regions changed')
    if sha256_file(args.source_regions) != protocol['full_source_regions_sha256']:
        raise ValueError('Full frozen source regions changed')
    wanted = {case['evidence_id'] for case in protocol['cases']}
    overlays = []
    with args.source_regions.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['evidence_id'] in wanted and row.get('extraction', {}).get('kind') == KIND:
                overlays.append(row)
    if len(overlays) != 2:
        raise ValueError('Expected two existing native source overlays')
    browser = EvidenceBrowser(args.audit, args.phase, data_archive=args.data_archive,
        cache_dir=args.output/'original-pdf-cache')
    if browser.audit['source_archives']['data']['sha256'] != protocol['data_archive_sha256']:
        raise ValueError('Frozen source archive does not match protocol')
    visual = VisualEvidence(SourceVerifier(browser, overlays))
    results = []
    for case in protocol['cases']:
        target = args.output/case['case']
        binding = visual.prepare_region(case['evidence_id'], target, dpi=protocol['dpi'])
        verification = visual.verify(target)
        checks = {'original_pdf_page_region_binding': verification['all_passed'],
                  'document_name': binding['source']['document_name'] == case['document_name'],
                  'page': binding['source']['page'] == case['page'],
                  'expected_selection': binding['selection'] == case['expected_selection'],
                  'no_invented_bbox': binding['source_geometry_fabricated'] is False,
                  'no_visual_inference_claim': binding['vision_reasoning_performed'] is False}
        results.append({'case': case['case'], 'source': binding['source'],
            'source_bbox_original': binding['source_bbox_original'], 'selection': binding['selection'],
            'fallback_reason': binding['fallback_reason'], 'binding_id': binding['binding_id'],
            'page_image': binding['page_image'], 'crop_image': binding['crop_image'],
            'verification': verification, 'checks': checks, 'all_passed': all(checks.values())})
    crm = [r for r in results if r['source']['document_name'] == 'crm_4502338']
    cross_page = {'same_original_pdf_sha256': len(crm) == 2 and crm[0]['source']['doc_sha256'] == crm[1]['source']['doc_sha256'],
                  'different_page_idx': len(crm) == 2 and crm[0]['source']['page_idx'] != crm[1]['source']['page_idx'],
                  'different_page_png_sha256': len(crm) == 2 and crm[0]['page_image']['sha256'] != crm[1]['page_image']['sha256']}
    files = {str(p.relative_to(args.output)).replace('\\', '/'): {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
             for p in sorted(args.output.rglob('*')) if p.is_file()}
    return {'state': 'source_binding_diagnostic_passed' if all(r['all_passed'] for r in results) and all(cross_page.values()) else 'source_binding_diagnostic_failed',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'working_tree_clean': not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(),
        'protocol_sha256': sha256_file(args.protocol),
        'frozen_inputs': {str(args.audit/'regions.jsonl'): protocol['original_regions_sha256'],
                          str(args.source_regions): protocol['full_source_regions_sha256'],
                          str(args.data_archive): protocol['data_archive_sha256']},
        'cases': results, 'cross_page_checks': cross_page,
        'actual_original_documents': 2, 'actual_original_pages': 3,
        'actual_page_pngs': 3, 'actual_crop_pngs': 2, 'full_page_geometry_fallbacks': 1,
        'actual_pdf_page_renders': visual.render_calls, 'render_note': 'Three preparation and three independent original-PDF replay renders',
        'elapsed_seconds': time.perf_counter()-started,
        'qa_labels_opened': False, 'new_dependencies_installed': False, 'network_calls_allowed': False,
        'model_loading_calls': 0, 'model_generation_calls': 0, 'images_uploaded': 0,
        'vision_reasoning_performed': False, 'visual_token_budget_verified': False,
        'answer_accuracy_measured': False, 'supported_answer': False,
        'scope': 'Page/crop provenance and coordinates only; not visual reasoning or answer correctness',
        'files': files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('configs/visual-evidence-v1.json'))
    for name in ('audit', 'phase', 'data-archive', 'source-regions', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory; earlier artifacts are preserved')
    args.output.mkdir(parents=True, exist_ok=False)
    network_attempts = 0
    def blocked(*_, **__):
        nonlocal network_attempts
        network_attempts += 1
        raise RuntimeError('Network disabled for this local PDF diagnostic')
    try:
        with patch.object(socket.socket, 'connect', blocked), patch.object(socket, 'create_connection', blocked):
            report = validate(args)
        code = 0 if report['state'] == 'source_binding_diagnostic_passed' else 1
    except Exception as exc:
        report = {'state': 'source_binding_diagnostic_blocked', 'error_class': type(exc).__name__,
                  'error': str(exc), 'traceback': traceback.format_exc(),
                  'vision_reasoning_performed': False, 'supported_answer': False}
        code = 2
    report['blocked_network_attempts'] = network_attempts
    write(args.output/'visual-evidence-report.json', report)
    print(json.dumps({k: report[k] for k in ('state', 'vision_reasoning_performed', 'supported_answer')}, ensure_ascii=False))
    return code


if __name__ == '__main__': raise SystemExit(main())
