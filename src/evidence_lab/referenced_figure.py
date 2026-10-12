"""Literal document figure links propose original pixels, never answer facts.

Only a unique caption in the retrieved document, on a physical page already
selected by retrieval, is eligible. No question IDs, answers or reference pages.
"""
import copy
import math
import re

from .safe_data import evidence_output


_REFERENCE = re.compile(r'\bfig(?:ure)?\.?\s+([1-9][0-9]{0,2})(?![0-9A-Za-z])', re.I)
_CAPTION = re.compile(r'^\s*fig(?:ure)?\.?\s+([1-9][0-9]{0,2})(?![0-9A-Za-z])', re.I)


def caption_number(text):
    match = _CAPTION.match(text)
    return match.group(1) if match else None


def linked_figure_source(hits, selected, canonical_sources):
    """Follow an explicit retrieved body reference to one complete figure.

    Multiple eligible figures or duplicate caption numbers are ambiguous and
    produce no proposal. A link is source acquisition, not semantic validation.
    """
    if not 1 <= len(hits) <= 10 or not 1 <= len(selected) <= 3:
        raise ValueError('Figure-link source count exceeds the retrieval budget')
    if any(source['type'] != 'text' for source in selected):
        return None, {'schema': 'literal_body_figure_link_acquisition-v1',
            'reason': 'Existing selected nontext region retains acquisition priority',
            'eligible_figures': [], 'maximum_new_figures': 1,
            'search_link_establishes_answer_support': False}
    primary = hits[0]['doc_sha256']
    scopes = {(s['doc_sha256'], s['page']) for s in selected}
    present = {s['evidence_id'] for s in selected}
    references = {}
    for hit in hits:
        if hit['doc_sha256'] != primary or hit['type'] != 'text':
            continue
        text = hit.get('text', '')
        if not isinstance(text, str) or len(text) > 24000:
            raise ValueError('Figure-link original text exceeds source budget')
        for match in _REFERENCE.finditer(text):
            references.setdefault(match.group(1), []).append({
                'evidence_id': hit['evidence_id'], 'doc_sha256': primary,
                'page': hit['page'], 'literal': match.group(0),
                'character_span': list(match.span()),
                'reference_is_answer_fact': False})
    captions = {}
    for row in canonical_sources:
        if row['doc_sha256'] != primary or row['type'] != 'image':
            continue
        number = caption_number(row.get('text', ''))
        if number in references:
            captions.setdefault(number, []).append(row)
    candidates, rejected = [], []
    for number, sources in sorted(captions.items()):
        if len(sources) != 1:
            rejected.append({'figure_number': number, 'reason': 'ambiguous_document_caption',
                             'evidence_ids': [s['evidence_id'] for s in sources]})
            continue
        source = sources[0]
        if source['evidence_id'] in present or (primary, source['page']) not in scopes:
            continue
        candidates.append({'route': 'literal_retrieved_body_figure_link_v1',
            'canonical_source': copy.deepcopy(evidence_output(source)),
            'figure_number': number, 'literal_body_references': references[number],
            'acquisition_reason': 'Explicit body figure reference and unique original caption on an already selected page',
            'semantic_support_established': False, 'reference_pages_used': False,
            'generated_visual_hints_used': False})
    report = {'schema': 'literal_body_figure_link_acquisition-v1',
        'body_references': references, 'eligible_figures': [c['canonical_source']['evidence_id'] for c in candidates],
        'rejected_ambiguities': rejected, 'maximum_new_figures': 1,
        'unique_eligible_figure': len(candidates) == 1,
        'search_link_establishes_answer_support': False}
    return (candidates[0] if len(candidates) == 1 else None), report


def _native_lines(words):
    rows = []
    for word in sorted(words, key=lambda w: (w['top'], w['x0'])):
        if not rows or abs(word['top'] - rows[-1][0]) > 3:
            rows.append([word['top'], [word]])
        else:
            rows[-1][1].append(word)
    return [sorted(row, key=lambda w: w['x0']) for _, row in rows]


def linked_figure_window(question, source, words, width, height):
    """Keep the full original figure and adjacent native caption/context.

    Geometry follows the independently measured original-region primitive.
    Unlike its old broad use, this view requires an explicit linked source and
    an actual matching native caption before it can enter a recovery input.
    """
    from .readable_region import valid_box, native_word_lines
    if not isinstance(question, str) or not question.strip() or len(question) > 500:
        raise ValueError('Linked-figure question exceeds source budget')
    box = source.get('bbox_points_estimate')
    number = caption_number(source.get('text', ''))
    if (source.get('type') != 'image' or number is None
            or not valid_box(box, width, height)
            or source.get('bbox_coordinate_system') != 'MinerU integer 0..1000, top-left; PDF points estimate is quantized'):
        raise ValueError('Linked figure needs its canonical caption and complete original geometry')
    area = (box[2]-box[0])*(box[3]-box[1])/(width*height)
    if not .01 <= area <= .80 or len(words) > 15000:
        raise ValueError('Linked figure exceeds original-region budget')
    nearby = [w for w in words if w['x1'] >= box[0]-18 and w['x0'] <= box[2]+18
              and w['bottom'] >= box[1]-48 and w['top'] <= box[3]+48
              and not (w['x0'] >= box[0] and w['x1'] <= box[2]
                       and w['top'] >= box[1] and w['bottom'] <= box[3])]
    identities = {(w['text'], w['x0'], w['top'], w['x1'], w['bottom']) for w in nearby}
    viewport = [max(0, box[0]-18), max(0, box[1]-18),
                min(width, box[2]+18), min(height, box[3]+18)]
    context, matching = [], []
    for line in _native_lines(words):
        if not any((w['text'], w['x0'], w['top'], w['x1'], w['bottom']) in identities for w in line):
            continue
        text = ' '.join(w['text'] for w in line)
        extent = [min(w['x0'] for w in line), min(w['top'] for w in line),
                  max(w['x1'] for w in line), max(w['bottom'] for w in line)]
        if not valid_box(extent, width, height):
            raise ValueError('Linked-figure caption/context lies outside the original page')
        record = {'text': text, 'bbox_top_left_points': extent,
            'native_word_records': [{k: w[k] for k in ('text','x0','top','x1','bottom')} for w in line],
            'is_answer_sentence': False}
        context.append(record)
        if caption_number(text) == number:
            matching.append(record)
        viewport = [max(0, min(viewport[0], extent[0]-6)), max(0, min(viewport[1], extent[1]-6)),
                    min(width, max(viewport[2], extent[2]+6)), min(height, max(viewport[3], extent[3]+6))]
    if len(matching) != 1:
        raise ValueError('Linked figure lacks one matching adjacent original native caption')
    visible = [w for w in words if w['x0'] >= viewport[0] and w['x1'] <= viewport[2]
               and w['top'] >= viewport[1] and w['bottom'] <= viewport[3]]
    pixels144 = math.ceil((viewport[2]-viewport[0])*2)*math.ceil((viewport[3]-viewport[1])*2)
    return {'status': 'selected_native_pdf_window', 'policy': 'literal_link_complete_original_figure_v1',
        'question': question, 'kind': 'original_quantized_figure_region', 'referenced_figure_context': True,
        'full_page_recovery': False, 'native_object_bbox_top_left_points': list(box),
        'viewport_bbox_top_left_points': viewport, 'margin_pdf_points': 18,
        'source_bbox_precision': 'Original quantized figure extent, not verified graph nodes or cells',
        'original_context_lines': context, 'matching_native_caption': matching[0],
        'native_word_records': [{k: w[k] for k in ('text','x0','top','x1','bottom')} for w in visible],
        'native_visible_text': ' '.join(w['text'] for w in visible),
        'native_context_text': ' '.join(c['text'] for c in context),
        'native_spatial_word_lines': native_word_lines(visible),
        'spatial_word_groups_are_semantic_cells': False, 'dpi': 144 if pixels144 <= 1_350_000 else 100,
        'entire_original_figure_extent_retained': True, 'association_verified': False,
        'fact_localization_verified': False, 'table_continuity_verified': False,
        'source_caption_is_answer_sentence': False, 'reference_pages_used': False}
