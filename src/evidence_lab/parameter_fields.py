"""Literal parameter rows, with independent word/label/header geometry checks.

This produces a source-field candidate, never a model quote or support verdict.
No document IDs, question IDs, reference pages or benchmark values occur here.
"""
import re

from .native_fields import SCALAR, box, word_rows
from .readable_region import valid_box

SUBSCRIPTS = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')


def requested_labels(question):
    if not isinstance(question, str) or not re.search(r'\b(?:values?|measurements?|estimates?)\b', question, re.I):
        return []
    if re.search(r'\b(?:difference|change|ratio|units?)\b|\b(?:19|20)\d{2}\b', question, re.I):
        return []
    normalized = question.translate(SUBSCRIPTS)
    labels = list(dict.fromkeys(greek + digit for greek, digit in
                  re.findall(r'(?<!\w)([α-ωΑ-Ω])\s*_?\s*(\d{1,2})(?!\d)', normalized)))
    return labels if 2 <= len(labels) <= 6 else []


def at_box(words, rectangle):
    if not isinstance(rectangle, list) or len(rectangle) != 4:
        return None
    matches = [w for w in words if box(w) == rectangle]
    return matches[0] if len(matches) == 1 else None


def checked_row(row, words, width, height):
    labels, columns, values = row.get('row_label_boxes', []), row.get('columns', []), row.get('values', [])
    headers, cells = row.get('header_boxes', []), row.get('value_boxes', [])
    if not 3 <= len(columns) <= 16 or len(set(columns)) != len(columns):
        return None
    if len(headers) != len(columns) or len(cells) != len(values) or len(values) != len(columns) or not 1 <= len(labels) <= 3:
        return None
    if any(not valid_box(b, width, height) for b in labels + headers + cells):
        return None
    label_words = [at_box(words, b) for b in labels]
    header_words = [at_box(words, b) for b in headers]
    value_words = [at_box(words, b) for b in cells]
    if any(w is None for w in label_words + header_words + value_words):
        return None
    if ''.join(w['text'] for w in label_words) != row.get('row_label'):
        return None
    if [w['text'] for w in header_words] != columns or [w['text'] for w in value_words] != values:
        return None
    if any(not SCALAR.fullmatch(v) for v in values):
        return None
    if any(w['x1'] >= value_words[0]['x0'] for w in label_words):
        return None
    if max(w['top'] for w in value_words) - min(w['top'] for w in value_words) > 3:
        return None
    if any(abs(w['top'] - value_words[0]['top']) > 15 for w in label_words):
        return None
    if any(header_words[i]['x0'] >= header_words[i + 1]['x0'] or value_words[i]['x0'] >= value_words[i + 1]['x0']
           for i in range(len(columns) - 1)):
        return None
    centers = [(w['x0'] + w['x1']) / 2 for w in header_words]
    spacing = min(centers[i + 1] - centers[i] for i in range(len(centers) - 1))
    if spacing <= 0 or any(abs((w['x0'] + w['x1']) / 2 - center) > min(16, spacing / 4)
                           for w, center in zip(value_words, centers)):
        return None
    if max(w['top'] for w in header_words) >= min(w['top'] for w in value_words):
        return None
    headings = [(y, line) for y, line in word_rows(words)
                if header_words[0]['top'] - 180 <= y < header_words[0]['top']
                and line and line[0]['text'].casefold() == 'table']
    if not headings:
        return None
    title_y, title_words = headings[-1]
    if any(header_words[0]['top'] < y < value_words[0]['top'] and line and line[0]['text'].casefold() == 'table'
           for y, line in word_rows(words)):
        return None
    return {'row_label': row['row_label'], 'columns': list(columns), 'values': list(values),
            'row_label_words': label_words, 'column_header_words': header_words, 'value_words': value_words,
            'title': ' '.join(w['text'] for w in title_words), 'title_words': title_words,
            'title_top': title_y, 'model_or_dataset_inferred': False, 'units_inferred': False}


def requested_parameter_candidate(question, catalog):
    requested = requested_labels(question)
    if not requested:
        return None
    groups = []
    seen = set()
    for image_id, source in catalog.items():
        viewport = source.get('readable_viewport', {})
        if viewport.get('status') != 'selected_native_pdf_window':
            continue
        words = viewport.get('native_word_records', [])
        bounds = viewport.get('viewport_bbox_top_left_points')
        if not words or len(words) > 15000 or not isinstance(bounds, list) or len(bounds) != 4:
            continue
        rows = viewport.get('native_parameter_rows', [])
        selected = []
        for label in requested:
            matches = [r for r in rows if r.get('row_label') == label]
            if len(matches) != 1:
                return None
            checked = checked_row(matches[0], words, max(bounds[2], max(w['x1'] for w in words)),
                                  max(bounds[3], max(w['bottom'] for w in words)))
            if checked is None:
                return None
            selected.append(checked)
        if len({r['title'] for r in selected}) != 1 or len({tuple(r['columns']) for r in selected}) != 1:
            return None
        key = (source['doc_sha256'], selected[0]['title'])
        if key in seen:
            return None
        seen.add(key)
        groups.append({'image_id': image_id, 'doc_sha256': source['doc_sha256'], 'page': source['page'],
                       'source_image_sha256': source['image']['sha256'], 'title': selected[0]['title'], 'rows': selected})
    if not 1 <= len(groups) <= 3 or sum(len(r['values']) for g in groups for r in g['rows']) > 96:
        return None
    if sum(len(g['rows']) for g in groups) > 8:
        return None
    parts = []
    for group in groups:
        for row in group['rows']:
            parts.append(group['title'] + ': ' + row['row_label'] + ' row: ' + ', '.join(
                name + '=' + value for name, value in zip(row['columns'], row['values'])))
    answer = '; '.join(parts) + '.'
    if len(answer) > 2400:
        return None
    return {'schema': 'source-native-parameter-row-candidate-v1', 'answer': answer,
            'evidence_ids': [g['image_id'] for g in groups],
            'source_field_plan': {'schema': 'literal-row-column-title-fields-v1', 'requested_row_labels': requested,
                                  'groups': groups, 'literal_source_word_geometry_checked': True,
                                  'semantic_support_established': False},
            'origin': 'literal source parameter rows with title/column/word-coordinate binding; not Qwen Reader text',
            'semantic_support_established': False}
