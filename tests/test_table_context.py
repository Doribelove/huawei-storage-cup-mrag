"""Actual self-authored PDFs; context hints are not a semantic benchmark."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from reportlab.pdfgen import canvas

from evidence_lab.native_pdf import SourceVerifier, native_tables
from evidence_lab.safe_data import sha256_file
from evidence_lab.table_context import ContextBudget, TableContext, header_hints
from evidence_lab.visual_evidence import VisualEvidence


def draw_table(c, *, top, bottom, header=None, columns=3):
    edges = [50+i*500/columns for i in range(columns+1)]
    levels = [top-i*(top-bottom)/4 for i in range(5)]
    for x in edges: c.line(x, bottom, x, top)
    for y in levels: c.line(50, y, 550, y)
    c.setFont('Helvetica', 9)
    for row in range(4):
        values = header if row == 0 and header else ['Body '+chr(65+column) for column in range(columns)]
        for column, value in enumerate(values): c.drawString(edges[column]+7, levels[row]-18, value)


def fixture(root, *, variant='chain', primary_header=False, known_neighbors=False, caption=False):
    root = Path(root); path = root/'own.pdf'
    c = canvas.Canvas(str(path), pagesize=(612, 792))
    draw_table(c, top=400, bottom=50, header=['Item', 'Count', 'Value'])
    c.showPage()
    if variant != 'gap':
        draw_table(c, top=720, bottom=80,
            columns=2 if variant == 'different_columns' else 3,
            header=['Category', 'Value', 'Units'] if variant == 'unrelated_same_grid' else None)
        if variant == 'unrelated_same_grid': c.drawString(50, 750, 'Table 99: Different subject')
    c.showPage()
    draw_table(c, top=720, bottom=320 if variant == 'other_same_page_header' else 80,
        header=['Item', 'Count', 'Value'] if primary_header else None)
    if variant == 'other_same_page_header':
        draw_table(c, top=270, bottom=50, header=['Item', 'Count', 'Value'])
    if caption: c.drawString(50, 755, 'Table 1: Own diagnostic subject')
    c.showPage(); c.save()
    doc = {'sha256': sha256_file(path), 'member': 'own.pdf', 'page_count': 3}
    class Browser:
        audit = {'source_archives': {'data': {'sha256': 'own-diagnostic-only'}}}
        documents = {'own': doc}; byid = {}
        def checked_page(self, name, page):
            if name != 'own' or isinstance(page, bool) or not isinstance(page, int) or not 1 <= page <= 3:
                raise ValueError('Page outside own source')
            return doc
        def verified_pdf(self, name):
            if name != 'own': raise ValueError('Unknown document')
            return path
    browser = Browser()
    primary_row = native_tables(path, doc, 'own', 3, 'own-diagnostic-only')[0]
    rows = ([r for page in (1, 2, 3) for r in native_tables(path, doc, 'own', page, 'own-diagnostic-only')]
            if known_neighbors else [primary_row])
    visual = VisualEvidence(SourceVerifier(browser, rows))
    primary = root/'primary'; binding = visual.prepare_region(primary_row['evidence_id'], primary)
    return visual, primary, binding


class TableContextTests(unittest.TestCase):
    def test_two_page_chain_adds_verified_candidate_headers_and_preserves_primary(self):
        with tempfile.TemporaryDirectory() as root:
            visual, primary, _ = fixture(root)
            before = {p.name: sha256_file(p) for p in primary.iterdir()}
            context = TableContext(visual); target = Path(root)/'context'
            result = context.prepare(primary, target)
            self.assertEqual([r['page'] for r in result['candidate_context']], [2, 1])
            self.assertFalse(result['candidate_context'][0]['header_hints'])
            self.assertTrue(result['candidate_context'][1]['header_hints'])
            self.assertIsNone(result['same_page_context'])
            self.assertEqual(result['budget_usage']['unique_pdf_pages'], 3)
            self.assertLessEqual(result['budget_usage']['total_selected_pixels'], 3_000_000)
            for candidate in result['candidate_context']:
                self.assertTrue(candidate['source_verification']['all_passed'])
                self.assertEqual(candidate['role'], 'candidate_context')
                self.assertFalse(candidate['table_continuity_verified'])
                self.assertFalse(candidate['header_assigned_to_primary_table'])
            self.assertFalse(result['table_continuity_verified'])
            self.assertFalse(result['primary']['retrieval_ranking_read_or_modified'])
            self.assertTrue(context.verify(target)['all_passed'])
            self.assertEqual(before, {p.name: sha256_file(p) for p in primary.iterdir()})

    def test_primary_header_present_stops_without_page_expansion(self):
        with tempfile.TemporaryDirectory() as root:
            visual, primary, _ = fixture(root, primary_header=True)
            result = TableContext(visual).prepare(primary, Path(root)/'context')
            self.assertEqual(result['candidate_context'], [])
            self.assertEqual(result['stop_reason'], 'primary_already_contains_lexical_header_hint')
            self.assertEqual(result['budget_usage']['image_assets'], 1)

    def test_different_columns_and_gap_prevent_jumping_to_older_header(self):
        for variant in ('different_columns', 'gap'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as root:
                visual, primary, _ = fixture(root, variant=variant)
                result = TableContext(visual).prepare(primary, Path(root)/'context')
                self.assertEqual(result['candidate_context'], [])
                self.assertEqual(result['stop_reason'], 'layout_chain_stop')
                self.assertEqual(result['omissions'][0]['page'], 2)
                self.assertFalse((Path(root)/'context/candidate-page-1').exists())

    def test_unrelated_same_grid_header_is_not_semantically_attached(self):
        with tempfile.TemporaryDirectory() as root:
            visual, primary, _ = fixture(root, variant='unrelated_same_grid')
            result = TableContext(visual).prepare(primary, Path(root)/'context')
            self.assertEqual([r['page'] for r in result['candidate_context']], [2])
            candidate = result['candidate_context'][0]
            self.assertIn('Category', candidate['source_record']['text'])
            self.assertTrue(candidate['header_hints'])
            self.assertEqual(candidate['relation'], 'candidate_table_context_semantic_continuity_unverified')
            self.assertFalse(candidate['header_assigned_to_primary_table'])
            self.assertFalse(result['supported_answer'])

    def test_other_table_same_page_does_not_supply_primary_header(self):
        with tempfile.TemporaryDirectory() as root:
            visual, primary, _ = fixture(root, variant='other_same_page_header')
            result = TableContext(visual, ContextBudget(max_backward_pages=0)).prepare(primary, Path(root)/'context')
            self.assertFalse(result['primary_header_hints'])
            self.assertEqual(result['candidate_context'], [])
            self.assertEqual(result['stop_reason'], 'backward_page_limit_reached')

    def test_only_source_grounded_table_caption_can_add_same_page_context(self):
        with tempfile.TemporaryDirectory() as root:
            visual, primary, _ = fixture(root, caption=True)
            result = TableContext(visual, ContextBudget(max_backward_pages=0)).prepare(primary, Path(root)/'context')
            source = result['same_page_context']['native_surrounding_line_candidates'][0]
            self.assertIn('Table 1', source['source_text'])
            self.assertEqual(source['source_word_union_bbox_top_left_points'][0], 50)
            self.assertFalse(source['attribution_to_table_verified'])
            self.assertEqual(result['budget_usage']['image_assets'], 2)

    def test_pixel_budget_refuses_before_neighbor_render(self):
        with tempfile.TemporaryDirectory() as root:
            visual, primary, binding = fixture(root)
            pixels = math.prod(binding['selected_image']['size_pixels'])
            before = visual.render_calls
            result = TableContext(visual, ContextBudget(max_total_pixels=pixels+935_000-1)).prepare(primary, Path(root)/'context')
            self.assertEqual(result['candidate_context'], [])
            self.assertEqual(result['omissions'][0]['reason'], 'aggregate_pixel_budget')
            self.assertEqual(visual.render_calls-before, 1)  # original-primary replay only
            self.assertFalse((Path(root)/'context/candidate-page-2').exists())

    def test_page_image_limits_and_primary_too_large_fail_closed(self):
        for budget in (ContextBudget(max_unique_pages=2), ContextBudget(max_image_assets=1)):
            with tempfile.TemporaryDirectory() as root:
                visual, primary, _ = fixture(root)
                result = TableContext(visual, budget).prepare(primary, Path(root)/'context')
                self.assertLessEqual(result['budget_usage']['unique_pdf_pages'], budget.max_unique_pages)
                self.assertLessEqual(result['budget_usage']['image_assets'], budget.max_image_assets)
                self.assertEqual(result['stop_reason'], 'budget_stop')
        with tempfile.TemporaryDirectory() as root:
            visual, primary, _ = fixture(root)
            with self.assertRaises(ValueError): TableContext(visual, ContextBudget(max_total_pixels=1)).prepare(primary, Path(root)/'context')
            self.assertFalse((Path(root)/'context').exists())

    def test_existing_native_candidates_are_reused_without_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as root:
            visual, primary, _ = fixture(root, known_neighbors=True)
            result = TableContext(visual).prepare(primary, Path(root)/'context')
            self.assertEqual([r['page'] for r in result['candidate_context']], [2, 1])

    def test_continuity_and_image_tampering_are_rejected_by_replay(self):
        with tempfile.TemporaryDirectory() as root:
            visual, primary, _ = fixture(root); context = TableContext(visual); target = Path(root)/'context'
            context.prepare(primary, target)
            file = target/'table-context.json'; raw = file.read_bytes(); value = json.loads(raw)
            value['candidate_context'][0]['table_continuity_verified'] = True
            file.write_text(json.dumps(value), encoding='utf-8')
            self.assertFalse(context.verify(target)['all_passed'])
            file.write_bytes(raw)
            image = target/'candidate-page-2/page.png'; image.write_bytes(image.read_bytes()+b'tampered')
            with self.assertRaises(ValueError): context.verify(target)

    def test_primary_image_change_rejects_without_supplement_output(self):
        with tempfile.TemporaryDirectory() as root:
            visual, primary, _ = fixture(root)
            image = primary/'crop.png'; image.write_bytes(image.read_bytes()+b'tampered')
            with self.assertRaises(ValueError): TableContext(visual).prepare(primary, Path(root)/'context')
            self.assertFalse((Path(root)/'context').exists())

    def test_long_body_prose_and_repeated_hint_word_are_not_headers(self):
        self.assertEqual(header_hints([['Shipping logistics oversees incoming plant material requirements',
            'In house training on requirements for shipping plant material']]), [])
        self.assertEqual(header_hints([['Position duties', 'Position workflow']]), [])
        self.assertTrue(header_hints([['Position', 'SANC Responsibilities', 'Qualifications', 'Documentation']]))

    def test_policy_limits_cannot_be_silently_expanded(self):
        for kwargs in ({'max_backward_pages':3}, {'max_unique_pages':4}, {'max_image_assets':5},
                       {'max_total_pixels':3_000_001}, {'max_backward_pages':True}):
            with self.assertRaises(ValueError): ContextBudget(**kwargs)


if __name__ == '__main__': unittest.main()
