"""Local candidate interfaces; no claim that this is the organizer's final schema."""
import json
from pathlib import Path

from .safe_data import sha256_file


def configure_inventory(template, data_root, inventory):
    template, data_root, inventory = map(Path, (template, data_root, inventory))
    config = json.loads(template.read_text(encoding='utf-8'))
    summary = json.loads((inventory/'summary.json').read_text(encoding='utf-8'))
    if summary.get('schema') != 'question-independent-native-image-feature-inventory-v1':
        raise ValueError('Unsupported source-only inventory schema')
    features = inventory/'features.jsonl'
    expected = summary.get('image_denominator')
    if type(expected) is not int or expected < 0 or summary.get('completed_attempts') != expected:
        raise ValueError('Incomplete image inventory; failures must stay in the denominator')
    if (summary.get('corpus_sha256') != sha256_file(data_root/'data/full/regions.jsonl')
            or summary.get('original_audit_sha256') != sha256_file(data_root/'data/audit/data-audit.json')
            or summary.get('features_sha256') != sha256_file(features)):
        raise ValueError('Source/inventory binding mismatch')
    rows = [json.loads(line) for line in features.read_text(encoding='utf-8').splitlines()]
    if len(rows) != expected or len({row['evidence_id'] for row in rows}) != expected:
        raise ValueError('Inventory image coverage mismatch')
    if any(row.get('status') not in ('collected', 'failed') for row in rows):
        raise ValueError('Unfinished image inventory record')
    source_images = {row['evidence_id'] for line in (data_root/'data/full/regions.jsonl').read_text(encoding='utf-8').splitlines()
                     if (row := json.loads(line)).get('type') == 'image'}
    if {row['evidence_id'] for row in rows} != source_images:
        raise ValueError('Inventory does not cover every canonical source image')
    config['native_image_features_sha256'] = summary['features_sha256']
    return config


def final_citation_sources(citations, catalog):
    sources = []
    for citation in citations:
        image_id = citation.get('image_id')
        source = catalog.get(image_id)
        if not source or source.get('image_id') != image_id:
            raise ValueError('Final citation has no matching verified image source')
        entry = {
            'image_id': image_id,
            'document_name': source['document_name'],
            'page': source['page'],
            'pdf_sha256': source['doc_sha256'],
            'pdf_member': source['pdf_member'],
            'evidence_id': source['parent_evidence_id'],
            'image_sha256': source['image']['sha256'],
        }
        if citation.get('type') == 'text_quote':
            # Copy the final validated citation exactly; never generate a quote
            # or invent a pixel rectangle for a text citation.
            entry['quote'] = citation['quote']
        else:
            entry['region'] = citation['region']
        sources.append(entry)
    return sources


def export_final_answers(questions, batch):
    batch = Path(batch).resolve()
    summary = json.loads((batch/'summary.json').read_text(encoding='utf-8'))
    records = summary.get('cases', [])
    if (summary.get('question_denominator') != len(questions)
            or summary.get('completed_attempts') != len(questions)
            or len(records) != len(questions)):
        raise ValueError('Batch is incomplete; no rows may be dropped')
    by_id = {row['sample_id']: row for row in records}
    if len(by_id) != len(records) or set(by_id) != {q['sample_id'] for q in questions}:
        raise ValueError('Question/result coverage mismatch')
    results = []
    for question in questions:
        record = by_id[question['sample_id']]
        path = Path(record['result_file']).resolve()
        if not path.is_relative_to(batch):
            raise ValueError('Result file is outside this batch')
        actual = json.loads(path.read_text(encoding='utf-8'))
        if (not actual.get('completed') or actual.get('sample_id') != question['sample_id']
                or actual.get('question') != question['question']):
            raise ValueError('Result/question binding mismatch')
        released = actual.get('supported_answer') is True
        status = actual.get('status')
        answer = actual.get('answer')
        citations = actual.get('citations')
        sources = actual.get('citation_sources', [])
        if (type(answer) is not str or type(citations) is not list
                or status not in ('answered', 'not_enough_evidence', 'failed')):
            raise ValueError('Invalid final result contract')
        if released:
            if status != 'answered' or not answer.strip() or not citations:
                raise ValueError('Released result lacks answer/source binding')
            if (len(sources) != len(citations)
                    or any(source.get('image_id') != citation.get('image_id')
                           or type(source.get('page')) is not int or source['page'] < 1
                           or not source.get('document_name')
                           for source, citation in zip(sources, citations))):
                raise ValueError('Final source/page coverage mismatch')
        elif answer or citations or status == 'answered':
            raise ValueError('Withheld or failed candidates cannot be exported as final answers')
        elif sources:
            raise ValueError('Withheld result cannot export candidate sources')
        results.append({'sample_id': question['sample_id'], 'answer': answer,
                        'status': status, 'citations': citations, 'sources': sources})
    return {'schema': 'track1-local-candidate-output-v1',
            'official_schema_confirmed': False, 'question_count': len(questions),
            'results': results}
