"""Released benchmark interface, backed by the unchanged r58 source pipeline.

Only the HTTP question reaches inference. Benchmark labels and judge settings
are deliberately not imported here. No model download or environment install.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
import importlib.util
_dependency_spec = importlib.util.spec_from_file_location(
    '_mrag_release_dependencies', PROJECT/'rag-anything-experiments/dependency_loader.py')
_dependency_module = importlib.util.module_from_spec(_dependency_spec)
_dependency_spec.loader.exec_module(_dependency_module)
_dependency_module.configure_dependencies(PROJECT)
sys.path.insert(0, str(PROJECT / 'src'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.new')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def environment():
    from dotenv import dotenv_values
    config = dict(os.environ)
    name = config.get('DOTENV_FILE', str(PROJECT / '.env'))
    if Path(name).is_file():
        config.update({k: v for k, v in dotenv_values(name).items() if v is not None})
    return config


def path_setting(config, name, default=None):
    value = config.get(name) or default
    if not value:
        raise ValueError('Missing required setting: ' + name)
    result = Path(value).expanduser()
    return (PROJECT / result).resolve() if not result.is_absolute() else result.resolve()


def models(config):
    result = {}
    for key, alias in [('qwen', 'LLM_MODEL'), ('bge', 'EMBEDDING_MODEL'), ('reranker', 'RERANK_MODEL')]:
        path = path_setting(config, 'MRAG_' + key.upper(), config.get(alias))
        if not path.is_dir():
            raise ValueError('A local pinned model directory is required for ' + key)
        result[key] = path
    return result


def run_script(name, arguments, log, timeout):
    command = [sys.executable, '-I', '-B', str(PROJECT / 'scripts' / name), *map(str, arguments)]
    started = time.perf_counter()
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    with log.open('x', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=PROJECT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout)
    record = {'command': command, 'returncode': result.returncode,
              'seconds': time.perf_counter() - started, 'log': str(log)}
    save(log.with_suffix('.execution.json'), record)
    if result.returncode:
        raise RuntimeError('Stage failed; see ' + str(log))
    return record


def import_documents(pdf_root, parsed_root, output):
    """Build provenance from the actual supplied PDF/content-list bytes.

    This works with new documents; it does not read questions or answer files.
    Original archive identity is retained only for a byte-identical set of all
    supplied members, otherwise an explicitly typed member-manifest identity is
    used. No existing vectors or answer cache are imported.
    """
    from pypdf import PdfReader
    from evidence_lab.safe_data import document_name, normalized_bbox, record_text, validate_name
    pdf_root, parsed_root, output = map(lambda p: Path(p).resolve(), (pdf_root, parsed_root, output))
    if output.exists():
        raise ValueError('Import requires a new output directory')
    pdfs, parsed = {}, {}
    for root, pattern, dest, maximum in [(pdf_root, '*.pdf', pdfs, 32 * 1024**2),
                                          (parsed_root, '*_content_list.json', parsed, 2 * 1024**2)]:
        if not root.is_dir():
            raise ValueError('Missing input directory: ' + str(root))
        for path in sorted(root.rglob(pattern)):
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('Input member escapes its declared root')
            name = document_name(path.name)
            validate_name(name)
            if name in dest or path.stat().st_size > maximum:
                raise ValueError('Duplicate or oversized document member: ' + name)
            dest[name] = path
    if not 1 <= len(pdfs) <= 100 or set(pdfs) != set(parsed):
        raise ValueError('Require1..100 unambiguous PDF/content-list document pairs; no silent omissions')
    output.mkdir(parents=True)
    members, documents, all_records = [], {}, {}
    for name in sorted(pdfs):
        doc = None
        for kind, source, member in [
            ('pdf', pdfs[name], 'original_pdf_files_100/' + name + '_origin.pdf'),
            ('content_list', parsed[name], 'mineru-parsed/' + name + '/auto/' + name + '_content_list.json')]:
            target = output / 'data/original-members' / member
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            digest = sha(target)
            if digest != sha(source):
                raise ValueError('Source changed during copy')
            info = {'document_name': name, 'kind': kind, 'original_member': member,
                    'relative_path': str(target.relative_to(output)), 'bytes': target.stat().st_size,
                    'sha256': digest}
            members.append(info)
            if kind == 'pdf':
                reader = PdfReader(target, strict=False)
                pages = [{'width': float(p.mediabox.width), 'height': float(p.mediabox.height),
                          'rotation': int(p.get('/Rotate', 0)), 'cropbox': [float(x) for x in p.cropbox]}
                         for p in reader.pages]
                doc = {'member': member, 'bytes': info['bytes'], 'sha256': digest,
                       'page_count': len(pages), 'pages': pages}
            else:
                records = read(target)
                if not isinstance(records, list):
                    raise ValueError('Content list must be a JSON array')
                all_records[name] = records
                doc['content_list'] = {'member': member, 'sha256': digest, 'bytes': info['bytes'],
                                       'records': len(records)}
        documents[name] = doc
    provenance_members = sorted((m['original_member'], m['sha256'], m['bytes']) for m in members)
    identity = hashlib.sha256(json.dumps(provenance_members, ensure_ascii=False).encode()).hexdigest()
    identity_kind = 'sha256_of_actual_member_manifest_not_a_zip_hash'
    provenance = PROJECT / 'resources/provenance/source-members.json'
    if provenance.is_file():
        known = read(provenance)
        if sorted((m['original_member'], m['sha256'], m['bytes']) for m in known['members']) == provenance_members:
            identity = known['original_archive_sha256']
            identity_kind = 'provided_archive_identity_after_all_member_byte_equality'
    evidence, invalid = {}, []
    for name, records in all_records.items():
        doc = documents[name]
        body = hashlib.sha256()
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError('Content-list record is not an object')
            text = record_text(record)
            body.update((str(record.get('type')) + '\n' + text + '\n').encode())
            page = record.get('page_idx')
            try:
                bbox = normalized_bbox(record)
                if type(page) is not int or not 0 <= page < doc['page_count']:
                    raise ValueError('Page outside supplied PDF')
            except ValueError as exc:
                invalid.append({'document': name, 'record_index': index, 'reason': str(exc)})
                continue
            p = doc['pages'][page]
            eid = hashlib.sha256(json.dumps([doc['sha256'], page, record['bbox'], record.get('type'), text],
                                           ensure_ascii=False).encode()).hexdigest()
            alias = {'document_name': name, 'pdf_member': doc['member'],
                     'content_list_member': doc['content_list']['member'], 'record_index': index, 'image_member': None}
            if eid in evidence:
                evidence[eid]['aliases'].append(alias)
                continue
            evidence[eid] = {'role': 'source_pdf_region', 'evidence_id': eid, 'doc_sha256': doc['sha256'],
                'document_name': name, 'page': page + 1, 'page_idx': page, 'type': record.get('type'),
                'source_bbox': record['bbox'], 'bbox_normalized': bbox,
                'bbox_points_estimate': [bbox[0]*p['width'], bbox[1]*p['height'], bbox[2]*p['width'], bbox[3]*p['height']],
                'bbox_coordinate_system': 'MinerU integer 0..1000, top-left; PDF points estimate is quantized',
                'page_size_points': [p['width'], p['height']], 'page_rotation': p['rotation'],
                'text': text, 'image_member': None, 'source_archive_sha256': identity, 'aliases': [alias],
                'limits': ['native/OCR parsed source; no new visual inference performed']}
        doc['parsed_body_sha256'] = body.hexdigest()
    if not evidence:
        raise ValueError('No usable source records; invalid records retained in failed import directory')
    audit_dir = output / 'data/audit'
    audit_dir.mkdir(parents=True)
    corpus = audit_dir / 'regions.jsonl'
    corpus.write_bytes(''.join(json.dumps(v, ensure_ascii=False) + '\r\n' for v in evidence.values()).encode())
    save(audit_dir / 'data-audit.json', {'schema': 'fresh-document-directory-audit-v1',
        'evidence_corpus_sha256': sha(corpus), 'source_archives': {'data': {'sha256': identity, 'identity_kind': identity_kind}},
        'dataset': {'documents': documents, 'logical_pdf_files': len(documents),
                    'verified_total_pdf_pages': sum(d['page_count'] for d in documents.values()),
                    'invalid_regions': invalid, 'deduplicated_evidence_regions': len(evidence)},
        'questions_or_gold_read': False, 'precomputed_index_used': False})
    save(output / 'data/source-members.json', {'schema': 'original-member-relocation-v1',
        'original_archive_sha256': identity, 'identity_kind': identity_kind,
        'document_count': len(documents), 'members': members})
    return {'documents': len(documents), 'regions': len(evidence), 'invalid_regions': len(invalid),
            'identity_kind': identity_kind, 'audit_sha256': sha(audit_dir / 'data-audit.json')}


def build(config):
    model = models(config)
    storage = path_setting(config, 'RAG_STORAGE')
    if storage.exists():
        raise ValueError('RAG_STORAGE must be a new directory; previous builds are preserved')
    started = time.perf_counter()
    storage.mkdir(parents=True)
    source = storage / 'source'
    report = {'schema': 'r58-released-build-v1', 'status': 'running', 'stages': [], 'questions_or_gold_read': False}
    save(storage / 'build-adapter.json', report)
    try:
        report['import'] = import_documents(path_setting(config, 'DOCS_PATH', 'original_pdf_files_100'),
                                             path_setting(config, 'DOC_FILE_PATH', 'mineru-parsed'), source)
        scripts = [
            ('prepare_extracted_corpus.py', ['--data-root', source, '--model-dir', model['bge'],
                '--expected-document-count', report['import']['documents'], '--output', source/'data/full'], 600),
            ('build_full_index.py', ['--freeze-dir', source/'data/full', '--model-dir', model['bge'],
                '--original-corpus', source/'data/audit/regions.jsonl', '--expected-document-count', report['import']['documents'],
                '--device', config.get('MRAG_BUILD_DEVICE', 'cuda'), '--max-build-seconds', 7200], 7350),
            ('collect_native_image_features.py', ['--project', PROJECT, '--data-root', source,
                '--output', storage/'native-image-inventory'], 3600),
            ('configure_inventory.py', ['--template', PROJECT/'configs/agx-source-facts-v14.json',
                '--data-root', source, '--inventory', storage/'native-image-inventory', '--output', storage/'run-config.json'], 120)]
        for name, arguments, timeout in scripts:
            report['stages'].append(run_script(name, arguments, storage/'logs'/(name+'.log'), timeout))
            save(storage / 'build-adapter.json', report)
        report['status'] = 'completed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        save(storage / 'build-adapter.json', report)
    results = path_setting(config, 'BENCHMARK_NAME', 'r58_benchmark').with_name(
        path_setting(config, 'BENCHMARK_NAME', 'r58_benchmark').name + '_results')
    timing = results / 'timings.json'
    existing = read(timing) if timing.exists() else {}
    existing.update(build_time=round(report['elapsed_seconds'], 4),
        build_time_scope='Actual PDF/member audit, copying,source construction,embedding,index persistence,native image inventory; excludes prior MinerU parsing')
    save(timing, existing)
    print(json.dumps({'status': report['status'], 'build_time': report['elapsed_seconds']}, ensure_ascii=False), flush=True)


class QueryRunner:
    def __init__(self, config):
        self.config = config
        self.storage = path_setting(config, 'RAG_STORAGE')
        if read(self.storage/'build-adapter.json')['status'] != 'completed':
            raise ValueError('A completed build is required')
        self.model = models(config)
        self.output = path_setting(config, 'MRAG_RUN_OUTPUT', str(self.storage/'query-runs'/str(time.time_ns())))
        self.output.mkdir(parents=True, exist_ok=False)
        self.count = 0
        self.total = 0.0

    def answer(self, question):
        if not isinstance(question, str) or not question.strip() or len(question) > 20000:
            raise ValueError('question must be a nonempty string of at most20000 characters')
        started = time.perf_counter()
        self.count += 1
        sid = 'query_%06d' % self.count
        folder = self.output / sid
        folder.mkdir()
        save(folder/'request.json', {'question': question})
        # Preserve the qualified core's500-character budget rather than truncating
        # an evaluator question or silently using its prefix as the full question.
        if len(question) > 500:
            packet = {'answer': '', 'status': 'withheld', 'reason': 'qualified_core_question_limit_500', 'citations': []}
        else:
            save(folder/'questions.json', {'version': 'question-only-v1', 'questions': [{'sample_id': sid, 'question': question}]})
            try:
                run_script('run_submission.py', ['--data-root', self.storage/'source',
                    '--model-dir', self.model['qwen'], '--bge-dir', self.model['bge'], '--reranker-dir', self.model['reranker'],
                    '--native-image-inventory', self.storage/'native-image-inventory', '--config', self.storage/'run-config.json',
                    '--questions', folder/'questions.json', '--output', folder/'batch', '--max-run-seconds', 1800], folder/'inference.log', 1830)
                summary = read(folder/'batch/summary.json')
                result = read(summary['cases'][0]['result_file'])
                released = result.get('supported_answer') is True and result.get('status') == 'answered'
                packet = {'answer': result.get('answer', '') if released else '', 'status': result['status'],
                          'citations': result.get('citations', []) if released else []}
            except Exception as exc:
                packet = {'answer': '', 'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc), 'citations': []}
        elapsed = time.perf_counter() - started
        self.total += elapsed
        packet.update(query_time_s=round(elapsed, 4), avg_query_time_s=round(self.total/self.count, 4),
                      total_queries=self.count, request_id=sid)
        save(folder/'response.json', packet)
        return packet


def make_handler(runner):
    from http.server import BaseHTTPRequestHandler
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Never log bodies, environment variables, credentials or model paths.
            sys.stderr.write('%s %s\n' % (self.command, self.path))

        def respond(self, status, packet):
            raw = json.dumps(packet, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self.respond(200, {'status': 'ok'}) if self.path == '/health' else self.respond(404, {'error': 'not_found'})

        def do_POST(self):
            if self.path != '/query':
                return self.respond(404, {'error': 'not_found'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 128000:
                    raise ValueError('Request length exceeds limit')
                body = json.loads(self.rfile.read(size))
                # Labels, reference pages, sample IDs or selectors cannot enter
                # the answerer through this interface.
                if not isinstance(body, dict) or set(body) != {'question'}:
                    raise ValueError('Only the question field is accepted')
                self.respond(200, runner.answer(body['question']))
            except (ValueError, TypeError, UnicodeError) as exc:
                self.respond(422, {'error': str(exc)})
    return Handler


def serve(config):
    from http.server import HTTPServer
    import signal
    def stop_owned_server(signum, frame):
        raise KeyboardInterrupt('Owned server stopping')
    signal.signal(signal.SIGTERM, stop_owned_server)
    server = HTTPServer((config.get('RAG_IP', '127.0.0.1'), int(config.get('RAG_PORT', '8080'))),
                        make_handler(QueryRunner(config)))
    server.timeout = 1
    print(json.dumps({'status': 'listening', 'port': server.server_address[1], 'model_execution': 'serial_per_request',
                      'cold_model_load_included_in_query_time': True}), flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
