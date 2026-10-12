import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evidence_lab.referenced_figure import caption_number, linked_figure_source, linked_figure_window
from evidence_lab.source_recovery import recover_empty_reader


def source(eid, text, *, kind='text', page=1, doc='a'):
    return {'evidence_id': eid, 'text': text, 'type': kind, 'page': page,
            'page_idx': page-1, 'doc_sha256': doc, 'document_name': doc,
            'bbox_points_estimate': [100, 100, 300, 300],
            'bbox_coordinate_system': 'MinerU integer 0..1000, top-left; PDF points estimate is quantized'}


class ReferencedFigureTests(unittest.TestCase):
    def setUp(self):
        self.body = source('body', 'The controller is illustrated in Figure 12.')
        self.selected = [self.body, source('context', 'Continuation', page=2)]
        self.figure = source('fig', 'Figure 12. Controller layout.', kind='image', page=2)

    def test_unique_literal_source_link_preserves_original_and_offsets(self):
        before = copy.deepcopy(self.figure)
        proposal, report = linked_figure_source(self.selected, self.selected, [self.figure])
        self.assertTrue(report['unique_eligible_figure'])
        self.assertEqual(proposal['canonical_source'], self.figure)
        ref = proposal['literal_body_references'][0]
        lo, hi = ref['character_span']
        self.assertEqual(self.body['text'][lo:hi], ref['literal'])
        self.assertFalse(proposal['semantic_support_established'])
        proposal['canonical_source']['text'] = 'untrusted mutation'
        self.assertEqual(self.figure, before)

    def test_exact_numbers_do_not_match_substrings_panels_or_plural_ranges(self):
        self.assertEqual(caption_number('Fig. 12. Layout'), '12')
        for text in ('Figure 1201', 'Figure 12a: Panel', 'Figures 12 and 13', 'A Figure 12 caption'):
            self.assertIsNone(caption_number(text))
        other = source('other', 'Figure 2. Other layout', kind='image', page=2)
        proposal, _ = linked_figure_source(self.selected, self.selected, [other])
        self.assertIsNone(proposal)

    def test_cross_document_and_new_physical_pages_are_not_selected(self):
        for field, value in (('doc_sha256', 'b'), ('page', 3)):
            row = dict(self.figure, **{field: value})
            proposal, _ = linked_figure_source(self.selected, self.selected, [row])
            self.assertIsNone(proposal)

    def test_duplicate_document_caption_is_ambiguous_even_outside_selected_pages(self):
        duplicate = dict(self.figure, evidence_id='duplicate', page=9)
        proposal, report = linked_figure_source(self.selected, self.selected, [self.figure, duplicate])
        self.assertIsNone(proposal)
        self.assertEqual(report['rejected_ambiguities'][0]['reason'], 'ambiguous_document_caption')

    def test_multiple_eligible_figures_do_not_guess(self):
        body = dict(self.body, text='Figure 12 and Figure 13 show alternatives.')
        second = source('second', 'Figure 13. Alternative', kind='image', page=2)
        proposal, report = linked_figure_source([body], self.selected, [self.figure, second])
        self.assertIsNone(proposal)
        self.assertEqual(len(report['eligible_figures']), 2)

    def test_existing_image_or_table_keeps_priority(self):
        for kind in ('image', 'table'):
            selected = [self.body, source('current', 'Current', kind=kind, page=2)]
            proposal, report = linked_figure_source(self.selected, selected, [self.figure])
            self.assertIsNone(proposal)
            self.assertIn('priority', report['reason'])

    def test_nonempty_answer_does_not_request_linked_acquisition(self):
        original = {'answer_candidate_present': True}
        with patch('evidence_lab.referenced_figure.linked_figure_source') as search:
            returned, _, report = recover_empty_reader('Describe the diagram', original,
                SimpleNamespace(citation_catalog={}), hits=self.selected, selected=self.selected,
                visual=None, reader=None, processor=None, budget=None,
                config={'bounded_referenced_figure_recovery': True}, output=None)
        self.assertIs(returned, original)
        self.assertEqual(report['actual_reader_calls'], 0)
        search.assert_not_called()

    def caption_words(self, label='12.'):
        return [dict(text=t, x0=x, x1=x+30, top=310, bottom=320)
                for t, x in [('Figure', 100), (label, 138), ('Controller', 176)]]

    def test_whole_figure_and_actual_caption_coordinates_are_retained(self):
        words = self.caption_words()
        plan = linked_figure_window('Describe the controller', self.figure, words, 600, 800)
        self.assertTrue(plan['entire_original_figure_extent_retained'])
        self.assertEqual(plan['native_word_records'], words)
        self.assertEqual(plan['matching_native_caption']['text'], 'Figure 12. Controller')
        box = plan['viewport_bbox_top_left_points']
        self.assertLessEqual(box[0], 100)
        self.assertLessEqual(box[1], 100)
        self.assertGreaterEqual(box[2], 300)
        self.assertGreaterEqual(box[3], 320)
        self.assertFalse(plan['matching_native_caption']['is_answer_sentence'])
        self.assertFalse(plan['fact_localization_verified'])

    def test_generated_or_other_caption_cannot_replace_native_caption(self):
        for words in ([], self.caption_words('13.')):
            with self.assertRaisesRegex(ValueError, 'matching adjacent original native caption'):
                linked_figure_window('Describe the controller', self.figure, words, 600, 800)


if __name__ == '__main__':
    unittest.main()
