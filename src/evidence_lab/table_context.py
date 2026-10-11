"""Bounded source-only table context. Layout hints never prove table continuity."""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import pdfplumber
import pypdfium2 as pdfium

from .native_pdf import KIND, SourceVerifier, native_tables
from .safe_data import sha256_file
from .visual_evidence import VisualEvidence, _PDFIUM_LOCK, json_hash

# Generic lexical hints, not a semantic header classifier or sample-specific labels.
HEADER_WORDS = frozenset('position responsibilities qualifications requirements documentation '
    'name item product units count value date category owner current estimated stores sales'.split())


def normalized(text):
    return ' '.join(re.findall(r'\w+', str(text).casefold()))


def header_hints(cells):
    result = []
    for index, row in enumerate(cells[:3]):
        matches = [{ 'column_idx': column, 'source_text': text,
                     'hint_words': sorted(set(normalized(text).split()) & HEADER_WORDS)}
                   for column, text in enumerate(row)
                   if text and len(normalized(text).split()) <= 6 and len(normalized(text)) <= 64]
        matches = [m for m in matches if m['hint_words']]
        if len(matches) >= 2 and len({word for match in matches for word in match['hint_words']}) >= 2:
            result.append({'row_idx': index, 'matched_columns': matches,
                           'meaning': 'Lexical header candidate, not semantic classification'})
    return result


@dataclass(frozen=True)
class ContextBudget:
    max_backward_pages: int = 2
    max_unique_pages: int = 3
    max_image_assets: int = 4
    max_total_pixels: int = 3_000_000

    def __post_init__(self):
        for name, lower, upper in [('max_backward_pages', 0, 2), ('max_unique_pages', 1, 3),
                                  ('max_image_assets', 1, 4), ('max_total_pixels', 1, 3_000_000)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError('Context budget exceeds bounded policy: '+name)


class TableContext:
    def __init__(self, visual: VisualEvidence, budget: ContextBudget = ContextBudget()):
        if type(visual) is not VisualEvidence or type(budget) is not ContextBudget:
            raise TypeError('Use the existing verified visual source and bounded context policy')
        self.visual = visual
        self.budget = budget

    def _inspect(self, name, number):
        path, doc, _, _, _, _ = self.visual._source(None, name, number)
        rows = native_tables(path, doc, name, number,
            self.visual.browser.audit['source_archives']['data']['sha256'])
        result = []
        with pdfplumber.open(path) as pdf:
            page = pdf.pages[number-1]
            tables = page.find_tables()
            for row in rows:
                table = tables[row['extraction']['table_index']]
                # Use an actual fully separated row, never infer missing grid boundaries.
                full_rows = [r.cells for r in table.rows if r.cells and all(c is not None for c in r.cells)]
                cells = max(full_rows, key=len) if full_rows else []
                edges = [cells[0][0]]+[c[2] for c in cells] if cells else []
                # Actual outside-table PDF words only; never attribute nearby prose
                # or another table's header merely because it shares this page.
                word_lines = {}
                for word in page.extract_words():
                    word_lines.setdefault(round(word['top'], 1), []).append(word)
                surroundings = []
                box = row['source_bbox']
                for words in word_lines.values():
                    words.sort(key=lambda word: word['x0'])
                    if normalized(words[0]['text']) not in ('source', 'table'): continue
                    line_box = [min(w['x0'] for w in words), min(w['top'] for w in words),
                                max(w['x1'] for w in words), max(w['bottom'] for w in words)]
                    gap = box[1]-line_box[3] if line_box[3] <= box[1] else line_box[1]-box[3]
                    if 0 <= gap <= 72 and line_box[0] < box[2] and line_box[2] > box[0]:
                        surroundings.append({'source_text': ' '.join(w['text'] for w in words),
                            'source_word_union_bbox_top_left_points': line_box,
                            'extraction': 'pdfplumber native words outside table, within 72 points',
                            'attribution_to_table_verified': False})
                result.append({'record': row, 'layout': {'column_edges_pdf_points': edges,
                    'source_bbox': row['source_bbox'], 'page_size_points': [page.width, page.height]},
                    'header_hints': header_hints(row['extraction']['cells']), 'surroundings': surroundings})
        if sha256_file(path) != doc['sha256']:
            raise ValueError('Original PDF changed while inspecting table layout')
        return result

    def _page_cost(self, name, number, dpi):
        path, _, _, _, _, _ = self.visual._source(None, name, number)
        with _PDFIUM_LOCK, pdfium.PdfDocument(path) as pdf:
            page = pdf[number-1]
            try:
                width, height = page.get_size()
                if not all(math.isfinite(x) and 0 < x <= 10000 for x in (width, height)):
                    raise ValueError('Unsupported context page size')
                size = [math.ceil(width*dpi/72), math.ceil(height*dpi/72)]
            finally:
                page.close()
        return size, size[0]*size[1]

    def _plan(self, primary_dir):
        primary_dir = Path(primary_dir).resolve()
        primary = json.loads((primary_dir/'binding.json').read_text(encoding='utf-8'))
        if not self.visual.verify(primary_dir)['all_passed']:
            raise ValueError('Primary visual source has changed')
        source = primary['source']; eid = source['region_evidence_id']
        row = self.visual.verifier.extra.get(eid)
        if row is None or row['extraction']['kind'] != KIND or not primary['crop_image']:
            raise ValueError('This minimal context policy requires a verified native table crop')
        if primary['render']['dpi_at_user_unit1'] != 100:
            raise ValueError('Keep original primary rendering; context policy is fixed at 100 DPI')
        name, number = source['document_name'], source['page']
        original = next((item for item in self._inspect(name, number) if item['record']['evidence_id'] == eid), None)
        if original is None: raise ValueError('Primary table is absent from original native extraction')
        selected = primary['selected_image']
        pixels = math.prod(selected['size_pixels']); assets = 1
        if pixels > self.budget.max_total_pixels:
            raise ValueError('Primary image alone exceeds aggregate context pixel budget')
        plan = {'kind': 'bounded_table_visual_context', 'primary_directory': str(primary_dir),
            'primary': {'role': 'primary_retrieved_region', 'source': source,
                'binding_id': primary['binding_id'], 'binding_manifest_sha256': sha256_file(primary_dir/'binding.json'),
                'selected_image': selected, 'source_bbox_original': primary['source_bbox_original'],
                'retrieval_ranking_read_or_modified': False, 'original_crop_preserved': True},
            'same_page_context': None, 'candidate_context': [], 'omissions': [],
            'primary_header_hints': original['header_hints'],
            'table_continuity_verified': False, 'header_assigned_to_primary_table': False,
            'vision_reasoning_performed': False, 'visual_token_budget_verified': False,
            'answer_generated': False, 'supported_answer': False,
            'policy': {**self.budget.__dict__, 'dpi': 100, 'column_edge_tolerance_points': 2.0,
                'current_table_top_fraction_max': 0.15, 'previous_table_bottom_fraction_min': 0.80,
                'direction': 'previous adjacent pages only; no gap jumping',
                'continuity_meaning': 'Layout support only; every cross-page relation remains a candidate'},
            'stop_reason': None}
        if original['header_hints']:
            plan['stop_reason'] = 'primary_already_contains_lexical_header_hint'
        else:
            page_pixels = math.prod(primary['page_image']['size_pixels'])
            if original['surroundings'] and assets+1 <= self.budget.max_image_assets and pixels+page_pixels <= self.budget.max_total_pixels:
                plan['same_page_context'] = {'role': 'same_page_context', 'source': source,
                    'binding_id': primary['binding_id'], 'image': primary['page_image'],
                    'reason': 'Expose original same-page title/source-line surroundings omitted by the crop',
                    'native_surrounding_line_candidates': original['surroundings'],
                    'caption_attributed_to_primary_table': False, 'other_tables_may_be_present': True}
                pixels += page_pixels; assets += 1
            elif original['surroundings']:
                plan['omissions'].append({'page': number, 'reason': 'same_page_context_exceeds_image_or_pixel_budget'})
            current = original
            for step in range(1, self.budget.max_backward_pages+1):
                previous = number-step
                if previous < 1:
                    plan['stop_reason'] = 'document_start'; break
                if step+1 > self.budget.max_unique_pages or assets+1 > self.budget.max_image_assets:
                    plan['omissions'].append({'page': previous, 'reason': 'unique_page_or_image_budget'})
                    plan['stop_reason'] = 'budget_stop'; break
                # Bound raster cost before inspecting tables or rendering the candidate page.
                size, cost = self._page_cost(name, previous, 100)
                if pixels+cost > self.budget.max_total_pixels:
                    plan['omissions'].append({'page': previous, 'reason': 'aggregate_pixel_budget', 'required_pixels': cost})
                    plan['stop_reason'] = 'budget_stop'; break
                layout = current['layout']; edges = layout['column_edges_pdf_points']
                if not edges or layout['source_bbox'][1] > 0.15*layout['page_size_points'][1]:
                    plan['stop_reason'] = 'current_table_lacks_top_edge_layout_support'; break
                try:
                    candidates = self._inspect(name, previous)
                except ValueError as exc:
                    plan['omissions'].append({'page': previous, 'reason': 'native_page_geometry_unsupported', 'detail': str(exc)})
                    plan['stop_reason'] = 'unsupported_previous_page'; break
                matches = []
                for candidate in candidates:
                    other = candidate['layout']; other_edges = other['column_edges_pdf_points']
                    if (other['page_size_points'] == layout['page_size_points'] and len(other_edges) == len(edges)
                            and edges and max(abs(a-b) for a,b in zip(edges, other_edges)) <= 2.0
                            and other['source_bbox'][3] >= 0.80*other['page_size_points'][1]):
                        matches.append(candidate)
                if len(matches) != 1:
                    plan['omissions'].append({'page': previous, 'reason': 'no_unique_aligned_page_edge_table', 'matching_tables': len(matches)})
                    plan['stop_reason'] = 'layout_chain_stop'; break
                candidate = matches[0]
                candidate_id = candidate['record']['evidence_id']
                known = self.visual.verifier.extra.get(candidate_id)
                if known is not None:
                    if known != candidate['record']: raise ValueError('Existing candidate source row differs')
                    proof = self.visual.verifier.verify(candidate_id)
                else:
                    extras = [*self.visual.verifier.extra.values(), candidate['record']]
                    proof = SourceVerifier(self.visual.browser, extras).verify(candidate_id)
                if not proof['all_passed']: raise ValueError('Candidate table native source binding failed')
                plan['candidate_context'].append({'role': 'candidate_context', 'page': previous,
                    'adjacent_to_page': number-step+1, 'source_record': candidate['record'],
                    'source_verification': proof, 'layout': candidate['layout'],
                    'column_edge_max_delta_points': max(abs(a-b) for a,b in zip(edges, candidate['layout']['column_edges_pdf_points'])),
                    'header_hints': candidate['header_hints'], 'expected_page_size_pixels': size,
                    'reason': 'Same verified PDF, adjacent page, aligned columns and page-edge table; header search',
                    'relation': 'candidate_table_context_semantic_continuity_unverified',
                    'table_continuity_verified': False, 'header_assigned_to_primary_table': False})
                pixels += cost; assets += 1; current = candidate
                if candidate['header_hints']:
                    plan['stop_reason'] = 'lexical_header_candidate_found_continuity_unverified'; break
            if plan['stop_reason'] is None:
                plan['stop_reason'] = 'backward_page_limit_reached'
        plan['budget_usage'] = {'unique_pdf_pages': 1+len(plan['candidate_context']),
            'image_assets': assets, 'total_selected_pixels': pixels,
            'meaning': 'Counts original crop, reused same-page image and whole candidate pages; not model tokens'}
        self.visual._source(None, name, number)  # reject PDF changes during planning
        return plan

    def _attach(self, plan, output, *, replay):
        for candidate in plan['candidate_context']:
            folder = Path(output)/('candidate-page-'+str(candidate['page']))
            if replay:
                if not self.visual.verify(folder)['all_passed']:
                    raise ValueError('Candidate page visual source changed')
                binding = json.loads((folder/'binding.json').read_text(encoding='utf-8'))
            else:
                binding = self.visual.prepare_page(plan['primary']['source']['document_name'], candidate['page'], folder)
            if (binding['source']['document_name'] != plan['primary']['source']['document_name']
                    or binding['source']['doc_sha256'] != plan['primary']['source']['doc_sha256']
                    or binding['source']['page'] != candidate['page']
                    or binding['page_image']['size_pixels'] != candidate['expected_page_size_pixels']):
                raise ValueError('Candidate page source or predicted raster budget differs')
            candidate['visual_binding'] = {'directory': folder.name, 'binding_id': binding['binding_id'],
                                           'source': binding['source'], 'image': binding['page_image']}
        plan['context_id'] = json_hash(plan)
        return plan

    def prepare(self, primary_dir: Path, output: Path):
        output = Path(output)
        if output.exists(): raise ValueError('Use a new context directory')
        plan = self._plan(primary_dir)
        output.mkdir(parents=True, exist_ok=False)
        result = self._attach(plan, output, replay=False)
        (output/'table-context.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
            allow_nan=False)+'\n', encoding='utf-8')
        return result

    def verify(self, output: Path):
        output = Path(output)
        stored = json.loads((output/'table-context.json').read_text(encoding='utf-8'))
        # Stored policy is not authoritative: recompute using this controlled policy.
        expected = self._attach(self._plan(Path(stored['primary_directory'])), output, replay=True)
        return {'context_id': stored['context_id'], 'all_passed': expected == stored,
                'meaning': 'Original source/layout/budget replay only; semantic table continuity remains unverified'}
