"""Deterministic native-PDF table evidence, separate from the frozen MinerU corpus.

No OCR, model, QA label, or inferred missing cell is used. Empty cells stay empty.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import pdfplumber
from .safe_data import evidence_output, reject_answer_fields, sha256_file

EXTRACTOR_VERSION = '0.11.9'
KIND = 'pdfplumber_native_table'


def native_tables(pdf_path: Path, document: dict, name: str, page_number: int,
                  archive_sha256: str) -> list[dict]:
    reject_answer_fields(document)
    if importlib.metadata.version('pdfplumber') != EXTRACTOR_VERSION:
        raise ValueError('Native PDF extractor version mismatch')
    if sha256_file(pdf_path) != document['sha256']:
        raise ValueError('Original PDF hash mismatch')
    if isinstance(page_number, bool) or not 1 <= page_number <= document['page_count']:
        raise ValueError('Original PDF page outside source')
    result = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        if page.rotation != 0 or list(page.cropbox) != list(page.mediabox):
            # Existing MinerU and native coordinate transforms differ on rotated/cropped pages.
            raise ValueError('Native table geometry needs an explicit crop/rotation transform')
        if len(page.edges) > 20000 or len(page.chars) > 100000:
            raise ValueError('Native page extraction budget exceeded')
        tables = page.find_tables()
        if len(tables) > 100:
            raise ValueError('Native table count budget exceeded')
        for table_index, table in enumerate(tables):
            cells = table.extract()
            if not cells or sum(bool(cell and cell.strip()) for row in cells for cell in row) < 2:
                continue
            if sum(len(row) for row in cells) > 10000:
                raise ValueError('Native table cell budget exceeded')
            text = '\n'.join('\t'.join(cell or '' for cell in row) for row in cells)
            if not text.strip() or len(text) > 100000:
                continue
            bbox = list(table.bbox)
            if not all(math.isfinite(v) for v in bbox) or not (
                    0 <= bbox[0] < bbox[2] <= page.width and 0 <= bbox[1] < bbox[3] <= page.height):
                raise ValueError('Native table bbox outside original page')
            extraction = {'kind': KIND, 'version': EXTRACTOR_VERSION,
                          'table_index': table_index, 'settings': 'pdfplumber-default-lines',
                          'cells': cells}
            binding = {'pdf_sha256': document['sha256'], 'document_name': name,
                       'page': page_number, 'bbox': bbox, 'text': text, 'extraction': extraction}
            eid = hashlib.sha256(json.dumps(binding, sort_keys=True, ensure_ascii=False,
                                           separators=(',', ':')).encode()).hexdigest()
            result.append({'role': 'source_pdf_region', 'evidence_id': eid,
                           'doc_sha256': document['sha256'], 'document_name': name,
                           'page': page_number, 'page_idx': page_number - 1, 'type': 'table',
                           'source_bbox': bbox,
                           'bbox_normalized': [bbox[0]/page.width, bbox[1]/page.height,
                                               bbox[2]/page.width, bbox[3]/page.height],
                           'bbox_points_estimate': bbox,
                           'bbox_coordinate_system': 'PDF native points; top-left; pdfplumber',
                           'page_size_points': [page.width, page.height], 'page_rotation': page.rotation,
                           'text': text, 'image_member': None,
                           'source_archive_sha256': archive_sha256,
                           'aliases': [{'document_name': name, 'pdf_member': document['member']}],
                           'extraction': extraction,
                           'limits': ['Original native PDF table strings, not visual or semantic verification',
                                      'Merged cells/reading order are extractor output; no inferred values']})
    return result


class SourceVerifier:
    """Verify old parsed evidence and native evidence against the original PDF archive."""
    def __init__(self, browser, extra_records: list[dict] | None = None):
        self.browser = browser
        self.extra = {}
        self.cache = {}
        for record in extra_records or []:
            row = evidence_output(record)
            if row.get('role') != 'source_pdf_region' or row.get('extraction', {}).get('kind') != KIND:
                raise ValueError('Only original native table overlays accepted')
            if row['evidence_id'] in self.extra or row['evidence_id'] in browser.byid:
                raise ValueError('Duplicate native evidence ID')
            if row['document_name'] not in browser.documents:
                raise ValueError('Unknown native source document')
            self.extra[row['evidence_id']] = row

    def verify(self, evidence_id: str) -> dict:
        if evidence_id not in self.extra:
            return self.browser.verify_citation(evidence_id)
        row = self.extra[evidence_id]
        document = self.browser.checked_page(row['document_name'], row['page'])
        pdf = self.browser.verified_pdf(row['document_name'])
        key = (row['document_name'], row['page'])
        if key not in self.cache:
            self.cache[key] = native_tables(pdf, document, row['document_name'], row['page'],
                                           self.browser.audit['source_archives']['data']['sha256'])
        original = next((r for r in self.cache[key]
                         if r['extraction']['table_index'] == row['extraction']['table_index']), None)
        checks = {'pdf_sha256': sha256_file(pdf) == row['doc_sha256'],
                  'page': original is not None and original['page'] == row['page'],
                  'source_bbox': original is not None and original['source_bbox'] == row['source_bbox'],
                  'normalized_bbox': original is not None and original['bbox_normalized'] == row['bbox_normalized'],
                  'text': original is not None and original['text'] == row['text'],
                  'extraction_binding': original is not None and evidence_output(original) == row}
        return {'evidence_id': evidence_id, 'checks': checks, 'all_passed': all(checks.values()),
                'source_kind': KIND, 'meaning': 'Original-PDF extraction binding; not answer correctness'}
