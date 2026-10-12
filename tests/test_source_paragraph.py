import unittest
from evidence_lab.source_paragraph import anchor_span,lexical_tokens,paragraph_window


class SourceParagraphTests(unittest.TestCase):
    def test_unique_boundary_anchors_keep_original_contiguous_positions(self):
        text='one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen'
        span=anchor_span(lexical_tokens(text),'one two three four five six seven eight\nnine ten eleven twelve thirteen fourteen fifteen sixteen')
        self.assertEqual((span['start_token'],span['end_token']),(0,16))
        self.assertFalse(span['entire_text_or_answer_semantics_verified'])
    def test_ambiguous_and_unknown_boundary_anchors_decline(self):
        text='one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen'
        self.assertIsNone(anchor_span(lexical_tokens(text+' '+text),text))
        self.assertIsNone(anchor_span(lexical_tokens(text),text+' missing boundary'))
    def test_line_hyphen_is_only_rejoined_across_actual_line_break(self):
        self.assertEqual(lexical_tokens('concen-\ntration'),['concentration'])
        self.assertEqual(lexical_tokens('low-concentration'),['low','concentration'])
    def test_parsed_math_font_spacing_is_not_a_missing_native_boundary(self):
        parsed=r'Results were stable for all observed groups at levels corresponding to $3 . 0 \mathrm { k g / m L }$ in total volume.'
        native='Results were stable for all observed groups at levels corresponding to 3.0 kg/mL in total volume.'
        original=parsed
        span=anchor_span(lexical_tokens(parsed),native)
        self.assertEqual(span['start_token'],0)
        self.assertEqual(span['end_token'],len(lexical_tokens(parsed)))
        self.assertEqual(parsed,original)
        self.assertFalse(span['entire_text_or_answer_semantics_verified'])
    def test_whole_native_paragraph_excludes_neighbor_caption_and_footer(self):
        words=[]
        for top,tokens in ((5,['Figure','Caption']),(30,['actual','paragraph']),(41,['continues','here']),(85,['footer'])):
            for i,text in enumerate(tokens):
                words.append({'text':text,'x0':10+i*30,'x1':35+i*30,'top':top,'bottom':top+8})
        source={'type':'text','bbox_points_estimate':[9,28,70,50]}
        plan=paragraph_window('Which facts?',source,words,100,100)
        self.assertEqual(plan['native_visible_text'],'actual paragraph\ncontinues here')
        self.assertEqual(len(plan['native_word_records']),4)
        self.assertTrue(plan['actual_native_words_replace_parsed_quote_text'])
        self.assertFalse(plan['fact_localization_verified'])
    def test_nontext_and_empty_native_region_cannot_masquerade_as_paragraph(self):
        for source in ({'type':'image','bbox_points_estimate':[0,0,50,50]},
                       {'type':'text','bbox_points_estimate':[0,0,50,50]}):
            with self.assertRaises(ValueError):paragraph_window('Question',source,[],100,100)

if __name__=='__main__':unittest.main()
