"""Source-only multimodal dry-run. Exit2 is expected blocked generation readiness."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import time
import traceback
from unittest.mock import patch

from evidence_lab.demo import EvidenceBrowser
from evidence_lab.lexical import BM25
from evidence_lab.native_pdf import KIND, SourceVerifier
from evidence_lab.reader import (LocalQwenReader, ReaderUnavailable, prepare_reader_visual_inputs,
                                 validate_visual_reader_output)
from evidence_lab.reader_context import PinnedReaderTokenizer, visual_reader_context_verified
from evidence_lab.safe_data import sha256_file
from evidence_lab.visual_evidence import VisualEvidence, json_hash


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


class ForbiddenCalls:
    """Call traps, not a simulated/verified processor or model."""
    def __init__(self): self.processor_calls = self.model_calls = 0
    def apply_chat_template(self, *_, **__):
        self.processor_calls += 1; raise RuntimeError('Actual multimodal processor use is outside this dry-run')
    def generate(self, *_, **__):
        self.model_calls += 1; raise RuntimeError('Model generation is forbidden in this dry-run')


def validate(args):
    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    previous = json.loads(args.previous_handoff.read_text(encoding='utf-8'))
    if previous['source_commit'] != protocol['base_commit']: raise ValueError('Use acceptedce24c18 handoff')
    previous_root = Path(previous['worktree'])
    def preserve_previous():
        for member, expected in previous['artifact_sha256'].items():
            if sha256_file(previous_root/member) != expected: raise ValueError('Accepted table-context artifact changed')
    preserve_previous()
    source_protocol = json.loads(Path('configs/visual-evidence-v1.json').read_text(encoding='utf-8'))
    if sha256_file(args.source_regions) != source_protocol['full_source_regions_sha256']:
        raise ValueError('Frozen full sources changed')
    if sha256_file(args.cached_crm_retrieval) != protocol['cached_crm_retrieval_sha256']:
        raise ValueError('Frozen actual CRM retrieval changed')
    rows = [json.loads(line) for line in args.source_regions.open(encoding='utf-8')]
    native = [row for row in rows if row.get('extraction', {}).get('kind') == KIND]
    browser = EvidenceBrowser(args.audit, args.phase, data_archive=args.data_archive,
        cache_dir=args.output/'original-pdf-cache')
    visual = VisualEvidence(SourceVerifier(browser, native))
    tokenizer = PinnedReaderTokenizer(args.tokenizer_dir)
    result = []; traps = ForbiddenCalls()
    reader = LocalQwenReader.__new__(LocalQwenReader)  # bypass model construction deliberately
    reader.processor = traps; reader.model = traps
    for case in protocol['cases']:
        folder = args.output/case['case']; folder.mkdir()
        if case['retrieval'].startswith('new_document_scoped'):
            scoped = [row for row in rows if row['document_name'] == case['document_name']]
            question = case['question']; hits = BM25(scoped).search(question, 5)
            provenance = {'kind': case['retrieval'], 'scope': case['document_name'], 'question': question,
                'source_corpus_sha256': source_protocol['full_source_regions_sha256'],
                'source_records_in_scope': len(scoped), 'new_bm25_queries': 1, 'new_model_inference': False}
        else:
            cached = json.loads(args.cached_crm_retrieval.read_text(encoding='utf-8'))['fixed_payload']
            question = cached['question']; hits = cached['hits']
            provenance = {'kind': case['retrieval'], 'source_artifact': str(args.cached_crm_retrieval.resolve()),
                'source_artifact_sha256': sha256_file(args.cached_crm_retrieval),
                'original_source_commit': '2c5166724dadaba503af4cd67dafa8caf4e0259a',
                'new_bm25_queries': 0, 'new_model_inference': False}
        before = copy.deepcopy(hits); before_sha = json_hash(hits)
        write(folder/'retrieval-input.json', {'provenance': provenance, 'question': question, 'hits': hits})
        eid = case['primary_evidence_id']
        prepared = prepare_reader_visual_inputs(question, hits, visual=visual,
            primary_artifacts={eid: args.primary_root/case['case']},
            table_context_artifacts={eid: args.context_root/case['case']}, tokenizer=tokenizer)
        guards = []
        for name, action in [('visual_read', lambda: reader.read_visual(prepared)),
                             ('text_read_rejects_visual_packet', lambda: reader.read(question, prepared))]:
            try:
                action(); guards.append({'route': name, 'blocked': False})
            except ReaderUnavailable as exc:
                guards.append({'route': name, 'blocked': True, 'reason': str(exc)})
        images = list(prepared.citation_catalog.values())
        diagnostic = prepared.report['budget_validation']['template_diagnostic']
        refusal = validate_visual_reader_output(json.dumps({'status':'not_enough_evidence','answer':'','citations':[]}), prepared)
        checks = {'actual_retrieval_input_unchanged': hits == before and json_hash(hits) == before_sha,
            'original_rank_preserved': prepared.report['primary_retrieval_ranks'] == [case['expected_primary_rank']],
            'expected_per_image_roles': [image['role'] for image in images] == case['expected_image_roles'],
            'expected_image_pages': [image['page'] for image in images] == case['expected_image_pages'],
            'source_packet_live_verified': visual_reader_context_verified(prepared),
            'actual_pinned_template_image_placeholders': diagnostic['image_placeholders'] == len(images),
            'no_placeholder_token_readiness': not prepared.report['ready_for_reader'] and not prepared.report['token_budget_verified'],
            'no_semantic_candidate_promotion': all(not image['table_continuity_verified'] and not image['header_assigned_to_primary_table'] for image in images),
            'both_reader_routes_blocked': all(guard['blocked'] for guard in guards),
            'output_contract_v2_refusal_presentation_empty': (refusal['output_contract_version'] == 2
                and refusal['status'] == 'not_enough_evidence' and refusal['answer'] == '' and refusal['citations'] == []
                and not refusal['supported_answer'] and refusal['presentation_status'] == 'reader_refused'),
            'structure_binding_do_not_establish_support': (refusal['structure_validation']['status'] == 'passed'
                and refusal['source_binding_validation']['status'] == 'passed'
                and not refusal['evidence_support']['semantic_verifier_run']),
            'processor_and_model_calls_zero': traps.processor_calls == traps.model_calls == 0}
        write(folder/'messages.json', prepared.messages)
        write(folder/'citation-catalog.json', prepared.citation_catalog)
        write(folder/'reader-stages.json', prepared.report)
        write(folder/'generation-guard.json', guards)
        write(folder/'synthetic-refusal-contract-check.json', {**refusal, 'diagnostic_only_not_a_model_answer': True})
        result.append({'case': case['case'], 'retrieval_provenance': provenance, 'retrieval_payload_sha256': before_sha,
            'original_primary_rank': prepared.report['primary_retrieval_ranks'][0],
            'image_roles': [image['role'] for image in images], 'image_pages': [image['page'] for image in images],
            'preparation': prepared.report['preparation'], 'budget_validation': prepared.report['budget_validation'],
            'generation': prepared.report['generation'], 'generation_guards': guards,
            'citation_trace': [{key: image[key] for key in ('image_id','role','parent_evidence_id','region_evidence_id',
                'candidate_native_evidence_id','doc_sha256','document_name','page','binding_id','image')} for image in images],
            'checks': checks, 'all_passed': all(checks.values())})
    preserve_previous()
    return {'state': 'visual_input_dry_run_passed_generation_blocked' if all(case['all_passed'] for case in result) else 'visual_input_diagnostic_failed',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'working_tree_clean': not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),
        'protocol_sha256': sha256_file(args.protocol), 'previous_handoff_sha256': sha256_file(args.previous_handoff),
        'accepted_context_artifacts_unchanged': True, 'actual_pinned_tokenizer_identity': tokenizer.identity,
        'cases': result, 'images_referenced': sum(case['preparation']['images'] for case in result),
        'new_images_materialized': 0, 'source_verification_render_calls': visual.render_calls,
        'native_multimodal_processor_calls': traps.processor_calls, 'model_loading_calls': 0, 'model_generation_calls': traps.model_calls,
        'new_model_inference': False, 'qa_labels_opened': False, 'supported_answer': False,
        'budget_validation_complete': False, 'pipeline_generation_complete': False,
        'elapsed_seconds': time.perf_counter()-started,
        'scope': 'Retrieval/source/context/local-image/messages/template/guard dry-run; no processor visual grid or answer inference'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('configs/reader-visual-dry-run-v1.json'))
    for name in ('audit','phase','data-archive','source-regions','cached-crm-retrieval','primary-root',
                 'context-root','previous-handoff','tokenizer-dir','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=False)
    attempts = 0
    def blocked(*_, **__):
        nonlocal attempts
        attempts += 1; raise RuntimeError('Network disabled for this local dry-run')
    try:
        with patch.object(socket.socket,'connect',blocked), patch.object(socket,'create_connection',blocked):
            report = validate(args)
        code = 2 if report['state'] == 'visual_input_dry_run_passed_generation_blocked' else 1
    except Exception as exc:
        report = {'state':'visual_input_preparation_error','error_class':type(exc).__name__,'error':str(exc),
            'traceback':traceback.format_exc(),'supported_answer':False}
        code = 3
    report['blocked_network_attempts'] = attempts
    report['readiness_exit_code'] = code
    report['exit_code_contract'] = '2=successful dry-run with expected generation block;1=diagnostic failure;3=preparation error;0 is unavailable until actual visual generation readiness is implemented'
    write(args.output/'reader-visual-dry-run-report.json', report)
    print(json.dumps({key:report[key] for key in ('state','readiness_exit_code','supported_answer')},ensure_ascii=False))
    return code


if __name__ == '__main__': raise SystemExit(main())
