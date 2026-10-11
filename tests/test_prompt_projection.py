import json
import unittest
from evidence_lab.prompt_projection import project_source_text


class PromptProjectionTests(unittest.TestCase):
    def test_cells_order_blanks_spans_entities_and_caption_survive(self):
        raw = '<table><tr><td></td><th>η &amp; β</th></tr><tr><td rowspan="2" colspan="3">0.03</td><td>-4.5</td></tr></table>\nCaption α'
        value, count = project_source_text(raw)
        self.assertEqual(count, 1)
        self.assertTrue(value.endswith('\nCaption α'))
        rows = json.loads(value[len('TABLE_CELL_ROWS '):].split('\nCaption')[0])
        self.assertEqual(rows, [['', 'η & β'], [{'text':'0.03','rowspan':2,'colspan':3}, '-4.5']])

    def test_unsupported_or_unbalanced_markup_is_preserved(self):
        for raw in ('<table><tr><td>7</tr></table>',
                    '<table><tr><td style="color:red">7</td></tr></table>',
                    '<table><tr><td><b>7</b></td></tr></table>'):
            self.assertEqual(project_source_text(raw), (raw, 0))

    def test_plain_source_is_never_summarized_or_selected(self):
        raw = 'A long source states 0.1, 0.2, units and limitations. No answer is inferred.'
        self.assertEqual(project_source_text(raw), (raw, 0))


if __name__ == '__main__': unittest.main()
