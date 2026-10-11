import copy
import unittest

from evidence_lab.native_fields import box
from evidence_lab.parameter_fields import requested_labels, requested_parameter_candidate
from evidence_lab.source_calculation import compose_source_parameter_candidate


def catalog(column_names=('Left', 'Middle', 'Right')):
    words = []
    def word(text, x, y, width=20):
        value = {'text': text, 'x0': x, 'x1': x + width, 'top': y, 'bottom': y + 10}
        words.append(value)
        return value
    for text, x in (('Table', 40), ('9:', 80), ('Source', 110), ('Alpha', 150)):
        word(text, x, 40)
    headers = [word(name, x, 100) for name, x in zip(column_names, (180, 260, 340))]
    rows = []
    for greek, y, values in (('δ', 200, ('1.25', '2.50', '3.75')), ('θ', 250, ('4.10', '5.20', '6.30'))):
        root, sub = word(greek, 60, y + 6, 6), word('2', 66, y + 9, 4)
        cells = [word(value, x, y) for value, x in zip(values, (180, 260, 340))]
        rows.append({'row_label': greek + '2', 'row_label_boxes': [box(root), box(sub)],
                     'columns': list(column_names), 'header_boxes': [box(w) for w in headers],
                     'values': list(values), 'value_boxes': [box(w) for w in cells]})
    return {'image-diagnostic': {'doc_sha256': 'a' * 64, 'page': 1, 'image': {'sha256': 'b' * 64},
                                'readable_viewport': {'status': 'selected_native_pdf_window',
                                   'viewport_bbox_top_left_points': [0, 20, 600, 400],
                                   'native_word_records': words, 'native_parameter_rows': rows}}}


class ParameterFieldsTests(unittest.TestCase):
    def test_requested_rows_bind_actual_titles_columns_and_literal_values(self):
        source = catalog()
        before = copy.deepcopy(source)
        candidate = requested_parameter_candidate('What are the values of δ2 and θ2?', source)
        self.assertIn('Table 9: Source Alpha', candidate['answer'])
        self.assertIn('δ2 row: Left=1.25, Middle=2.50, Right=3.75', candidate['answer'])
        self.assertIn('θ2 row: Left=4.10, Middle=5.20, Right=6.30', candidate['answer'])
        self.assertFalse(candidate['semantic_support_established'])
        self.assertEqual(len(candidate['source_field_plan']['groups'][0]['rows']), 2)
        self.assertEqual(source, before)

    def test_actual_permuted_column_names_follow_native_header_positions(self):
        candidate = requested_parameter_candidate('Give δ₂ and θ₂ values.', catalog(('Right', 'Left', 'Middle')))
        self.assertIn('δ2 row: Right=1.25, Left=2.50, Middle=3.75', candidate['answer'])

    def test_label_value_column_and_coordinate_tampering_each_abstain(self):
        for mutation in ('value', 'label', 'column', 'row_position'):
            source = catalog()
            row = source['image-diagnostic']['readable_viewport']['native_parameter_rows'][0]
            if mutation == 'value':
                row['values'][0] = '99.00'
            elif mutation == 'label':
                row['row_label_boxes'] = source['image-diagnostic']['readable_viewport']['native_parameter_rows'][1]['row_label_boxes']
            elif mutation == 'column':
                row['columns'][0] = 'Unseen'
            else:
                row['value_boxes'] = source['image-diagnostic']['readable_viewport']['native_parameter_rows'][1]['value_boxes']
                row['values'] = source['image-diagnostic']['readable_viewport']['native_parameter_rows'][1]['values']
            with self.subTest(mutation=mutation):
                self.assertIsNone(requested_parameter_candidate('Give δ2 and θ2 values.', source))

    def test_missing_native_title_and_duplicated_source_tables_abstain(self):
        source = catalog()
        source['image-diagnostic']['readable_viewport']['native_word_records'] = [
            w for w in source['image-diagnostic']['readable_viewport']['native_word_records'] if w['text'] != 'Table']
        self.assertIsNone(requested_parameter_candidate('Give δ2 and θ2 values.', source))
        source = catalog()
        source['duplicate'] = copy.deepcopy(source['image-diagnostic'])
        self.assertIsNone(requested_parameter_candidate('Give δ2 and θ2 values.', source))

    def test_incomplete_requested_row_or_extra_unresolved_field_abstain(self):
        for question in ('Give δ2 and ε2 values.', 'Give δ2 and θ2 values in 2024.',
                         'Give δ2 and θ2 values and their units.', 'Give the ratio of δ2 and θ2 values.'):
            self.assertIsNone(requested_parameter_candidate(question, catalog()))
        self.assertEqual(requested_labels('Give δ₂ and θ₂ measurements.'), ['δ2', 'θ2'])
        self.assertEqual(requested_labels('List process steps.'), [])

    def test_unregistered_context_cannot_compose_parameter_candidate(self):
        with self.assertRaises(ValueError):
            compose_source_parameter_candidate('Give δ2 and θ2 values.', {}, object())

    def test_two_titled_tables_three_rows_each_fit_actual_support_budget_without_omission(self):
        from evidence_lab.fact_support import candidate_claims
        source = catalog()
        viewport = source['image-diagnostic']['readable_viewport']
        row = copy.deepcopy(viewport['native_parameter_rows'][-1])
        row['row_label'] = 'ε2'
        for field in ('row_label_boxes', 'value_boxes'):
            for b in row[field]:
                b[1] += 50
                b[3] += 50
        for text, b in zip(('ε', '2'), row['row_label_boxes']):
            viewport['native_word_records'].append({'text': text, 'x0': b[0], 'top': b[1], 'x1': b[2], 'bottom': b[3]})
        for text, b in zip(row['values'], row['value_boxes']):
            viewport['native_word_records'].append({'text': text, 'x0': b[0], 'top': b[1], 'x1': b[2], 'bottom': b[3]})
        viewport['native_parameter_rows'].append(row)
        other = copy.deepcopy(source['image-diagnostic'])
        other['doc_sha256'] = 'c' * 64
        for word in other['readable_viewport']['native_word_records']:
            if word['text'] == 'Alpha':
                word['text'] = 'Beta'
        source['image-other'] = other
        candidate = requested_parameter_candidate('Give δ2, θ2 and ε2 values.', source)
        claims = candidate_claims(candidate['answer'])
        self.assertEqual(len(claims), 6)
        for group in candidate['source_field_plan']['groups']:
            for row in group['rows']:
                matching = [c['claim'] for c in claims if group['title'] in c['claim'] and row['row_label'] + ' row:' in c['claim']]
                self.assertEqual(len(matching), 1)
                for name, value in zip(row['columns'], row['values']):
                    self.assertIn(name + '=' + value, matching[0])


if __name__ == '__main__':
    unittest.main()
