"""Literal two-period, multirow table candidates with native word coordinates.

Ambiguous headings, units, periods or row coverage abstain. Candidates do not
establish support and never use benchmark IDs, scorer answers or reference pages.
"""
import re
from decimal import Decimal

from .native_fields import word_rows
from .source_semantics import MONTHS, NUMBER, decimal_value, normalized

_GENERIC = set('how did do does are is were was and or in on of at for the a an '
               'compared compare comparing to with differ difference differences '
               'change changes changed between from respective values value '
               'amount amounts during by as versus vs'.split())
_PERIOD = re.compile(r'^(?:\d+|[A-Za-z]+)\s+(?:months?|years?|quarters?)\s+ended\s+\d{1,2}\s+([A-Za-z]+)$', re.I)
_UNIT = re.compile(r"^(HK\$|US\$|\$|€|£|USD|EUR|GBP)(?:['’](000)|(['’])m)?$")
_TITLE = re.compile(r'^(?:Table\s+\S+|\d+[.])\s+\S+', re.I)


def _terms(text):
    result = set()
    for word in normalized(text).split():
        if word in _GENERIC or word.isdigit():
            continue
        if len(word) > 4 and word.endswith('ies'):
            word = word[:-3] + 'y'
        elif word.endswith(('xes', 'ches', 'shes', 'sses')):
            word = word[:-2]
        elif len(word) > 3 and word.endswith('s') and not word.endswith(('ss', 'us')):
            word = word[:-1]
        result.add(word)
    return result


def _center(word):
    return (word['x0'] + word['x1']) / 2


def _money(value, unit):
    match = _UNIT.fullmatch(unit)
    assert match
    scale = Decimal(1000 if match[2] else 1000000 if match[3] else 1)
    return match[1] + format(value * scale, ',f')


def requested_period_candidate(question, catalog):
    if (not isinstance(question, str) or len(question) > 1000 or
            not re.search(r'\b(differ|difference|change|compare|compared)\b', question, re.I) or
            re.search(r'\b(ratio|percent|percentage|proportion)\b', question, re.I)):
        return None
    years = re.findall(r'\b(?:19|20)\d{2}\b', question)
    if len(years) != 2 or len(set(years)) != 2:
        return None
    start, end = sorted(years)
    possibilities = []
    for image_id, source in catalog.items():
        viewport = source.get('readable_viewport', {})
        words = viewport.get('native_word_records', [])
        if not 1 <= len(words) <= 15000:
            continue
        # Only native words wholly inside the actual selected view qualify.
        bounds = viewport.get('viewport_bbox_top_left_points')
        if bounds is not None:
            words = [w for w in words if bounds[0] <= w['x0'] and w['x1'] <= bounds[2]
                     and bounds[1] <= w['top'] and w['bottom'] <= bounds[3]]
        rows = word_rows(words)
        headers = [(y, row) for y, row in rows if len(row) == 2
                   and {w['text'] for w in row} == set(years)]
        for hy, header in headers:
            centers = [_center(w) for w in header]
            gap = centers[1] - centers[0]
            if gap < 35:
                continue
            tolerance = min(16, gap / 4)
            units = [(y, row) for y, row in rows if hy + 3 <= y <= hy + 25
                     and len(row) == 2 and all(_UNIT.fullmatch(w['text']) for w in row)
                     and all(abs(_center(w)-c) <= tolerance for w, c in zip(row, centers))]
            if len(units) != 1 or units[0][1][0]['text'] != units[0][1][1]['text']:
                continue
            uy, unit_words = units[0]
            periods = [(y, row) for y, row in rows if hy - 40 <= y < hy - 3
                       and _PERIOD.fullmatch(' '.join(w['text'] for w in row))]
            if len(periods) != 1:
                continue
            py, period_words = periods[0]
            period = ' '.join(w['text'] for w in period_words)
            if (_PERIOD.fullmatch(period)[1].lower() not in MONTHS or
                    normalized(period) not in normalized(question)):
                continue
            titles = [(y, row) for y, row in rows if py - 100 <= y < py - 3
                      and _TITLE.match(' '.join(w['text'] for w in row))]
            if not titles:
                continue
            ty, title_words = titles[-1]
            title = ' '.join(w['text'] for w in title_words)
            q_without_period = normalized(question).replace(normalized(period), '')
            requested = _terms(q_without_period)
            if len(requested) < 2:
                continue
            next_header = min((y for y, _ in headers if y > hy), default=uy + 150)
            next_title = min((y for y, row in rows if y > uy and
                             _TITLE.match(' '.join(w['text'] for w in row))), default=uy + 150)
            observed = []
            for y, row in rows:
                if not uy + 3 <= y < min(uy + 150, next_header, next_title):
                    continue
                values = [w for w in row if NUMBER.fullmatch(w['text'])]
                if len(values) != 2:
                    continue
                if not all(abs(_center(w)-c) <= tolerance for w, c in zip(values, centers)):
                    continue
                labels = [w for w in row if w['x1'] < values[0]['x0'] - 10]
                if not labels or len(labels) + len(values) != len(row):
                    continue
                label = ' '.join(w['text'] for w in labels)
                matched = _terms(label) & requested
                if len(matched) < 2:
                    continue
                by_year = {h['text']: v for h, v in zip(header, values)}
                a, b = decimal_value(by_year[start]['text']), decimal_value(by_year[end]['text'])
                observed.append({'row_label':label, 'row_label_words':labels,
                    'column_years':[w['text'] for w in header], 'column_header_words':header,
                    'unit':unit_words[0]['text'], 'unit_words':unit_words,
                    'literal_values':[w['text'] for w in values], 'value_words':values,
                    'start_year':start, 'end_year':end, 'start_value':str(a), 'end_value':str(b),
                    'signed_change':str(b-a), 'operation':'end - start',
                    'matched_requested_terms':sorted(matched)})
            if not 2 <= len(observed) <= 4 or len({r['row_label'] for r in observed}) != len(observed):
                continue
            if not requested <= set().union(*(_terms(r['row_label']) for r in observed), _terms(title)):
                continue
            unit = unit_words[0]['text']
            parts = []
            for row in observed:
                delta = Decimal(row['signed_change'])
                direction = 'a decrease' if delta < 0 else 'an increase' if delta > 0 else 'no change'
                parts.append(f'For "{period}" under "{title}", row "{row["row_label"]}": '
                    f'{start} = {row["start_value"]} {unit}, {end} = {row["end_value"]} {unit}, '
                    f'change ({end} minus {start}) = {row["signed_change"]} {unit}, '
                    f'{direction} of {_money(abs(delta), unit)}')
            answer = '; '.join(parts) + '.'
            if len(answer) > 2400:
                continue
            plan = {'schema':'native-two-period-multirow-comparison-v1', 'image_id':image_id,
                'doc_sha256':source['doc_sha256'], 'page':source['page'],
                'source_image_sha256':source['image']['sha256'],
                'title':title, 'title_words':title_words, 'period':period, 'period_words':period_words,
                'rows':observed, 'arithmetic':'exact Decimal; source-literal unit scale',
                'semantic_support_established':False}
            possibilities.append({'schema':'source-native-period-comparison-candidate-v1',
                'answer':answer, 'evidence_ids':[image_id], 'source_field_plan':plan,
                'origin':'source-native two-period row comparison and Decimal arithmetic; not Qwen Reader text',
                'semantic_support_established':False})
    return possibilities[0] if len(possibilities) == 1 else None


def validate_period_candidate(question, validated, cited_catalog):
    supplied = validated.get('source_period_candidate')
    if supplied is None:
        return None
    checked = requested_period_candidate(question, cited_catalog)
    if checked is None or any(supplied.get(k) != checked[k] for k in
            ('schema', 'answer', 'evidence_ids', 'source_field_plan')):
        raise ValueError('Period comparison candidate differs from actual native fields')
    return checked['source_field_plan']
