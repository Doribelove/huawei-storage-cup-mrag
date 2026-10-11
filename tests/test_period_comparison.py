import copy
import unittest

from evidence_lab.period_comparison import requested_period_candidate, validate_period_candidate
from evidence_lab.source_calculation import compose_source_period_candidate
from evidence_lab.fact_support import candidate_claims


QUESTION = ('How did operating costs in North and South differ for the six months '
            'ended 30 June 2023 compared to 2022?')


def source(reverse=False, unit="€'000"):
    words = []
    def add(text, x, y, width=20):
        w = dict(text=text, x0=x, x1=x+width, top=y, bottom=y+10)
        words.append(w)
        return w
    for i, text in enumerate(('9.', 'Operating', 'costs')):
        add(text, 20+i*55, 40, 40)
    for i, text in enumerate(('Six', 'months', 'ended', '30', 'June')):
        add(text, 225+i*30, 75, 25)
    years = ('2022', '2023') if reverse else ('2023', '2022')
    for x, year in zip((300, 400), years):
        add(year, x, 100)
        add(unit, x, 115)
    for y, location, by_year in ((145, 'North', {'2022':'120', '2023':'80'}),
                                (165, 'South', {'2022':'20', '2023':'50'})):
        for i, text in enumerate(('Operating', 'costs', '-', location)):
            add(text, 20+i*55, y, 40)
        for x, year in zip((300, 400), years):
            add(by_year[year], x, y)
    return {'diagnostic-image':{'doc_sha256':'a'*64,'page':2,'image':{'sha256':'b'*64},
        'readable_viewport':{'native_word_records':words, 'viewport_bbox_top_left_points':[0,0,500,200]}}}


class PeriodComparisonTests(unittest.TestCase):
    def test_native_title_period_rows_units_and_decimal_directions(self):
        catalog = source()
        before = copy.deepcopy(catalog)
        candidate = requested_period_candidate(QUESTION, catalog)
        plan = candidate['source_field_plan']
        self.assertEqual(plan['title'], '9. Operating costs')
        self.assertEqual(plan['period'], 'Six months ended 30 June')
        self.assertEqual([r['signed_change'] for r in plan['rows']], ['-40','30'])
        self.assertIn('a decrease of €40,000', candidate['answer'])
        self.assertIn('an increase of €30,000', candidate['answer'])
        self.assertFalse(candidate['semantic_support_established'])
        self.assertEqual(catalog, before)
        self.assertLessEqual(len(candidate_claims(candidate['answer'])), 8)

    def test_swapped_native_year_columns_keep_year_value_binding(self):
        one = requested_period_candidate(QUESTION, source())
        two = requested_period_candidate(QUESTION, source(reverse=True))
        self.assertEqual(one['answer'], two['answer'])
        self.assertEqual(two['source_field_plan']['rows'][0]['column_years'], ['2022','2023'])

    def test_misaligned_value_or_unit_and_missing_units_abstain(self):
        for mutation in ('value', 'unit', 'missing', 'mismatch'):
            catalog = source()
            words = catalog['diagnostic-image']['readable_viewport']['native_word_records']
            if mutation == 'missing':
                words[:] = [w for w in words if w['top'] != 115]
            else:
                w = next(w for w in words if w['text'] == ('80' if mutation=='value' else "€'000"))
                if mutation == 'mismatch':
                    w['text'] = "$'000"
                else:
                    w['x0'] += 35; w['x1'] += 35
            self.assertIsNone(requested_period_candidate(QUESTION, catalog), mutation)

    def test_explicit_period_and_year_mismatch_abstain(self):
        for question in (QUESTION.replace('June','July'), QUESTION.replace('2023','2024'),
                         QUESTION.replace('six months ended 30 June','this period')):
            self.assertIsNone(requested_period_candidate(question, source()))

    def test_requested_row_coverage_and_total_are_not_substituted(self):
        for question in (QUESTION.replace('South','East'), QUESTION.replace('operating costs','total operating costs')):
            self.assertIsNone(requested_period_candidate(question, source()))
        catalog = source()
        words = catalog['diagnostic-image']['readable_viewport']['native_word_records']
        words[:] = [w for w in words if w['top'] != 165]
        self.assertIsNone(requested_period_candidate(QUESTION, catalog))

    def test_duplicate_bound_source_tables_are_ambiguous(self):
        catalog = source()
        catalog['duplicate'] = copy.deepcopy(catalog['diagnostic-image'])
        self.assertIsNone(requested_period_candidate(QUESTION, catalog))

    def test_new_title_prevents_cross_table_row_inheritance(self):
        catalog = source()
        words = catalog['diagnostic-image']['readable_viewport']['native_word_records']
        words.extend([dict(text=t,x0=x,x1=x+20,top=155,bottom=160)
                      for t,x in (('10.',20),('Another',70),('table',120))])
        self.assertIsNone(requested_period_candidate(QUESTION, catalog))

    def test_changed_native_values_recompute_and_zero_is_not_increase(self):
        catalog = source()
        next(w for w in catalog['diagnostic-image']['readable_viewport']['native_word_records']
             if w['text']=='80')['text'] = '120'
        candidate = requested_period_candidate(QUESTION, catalog)
        self.assertEqual(candidate['source_field_plan']['rows'][0]['signed_change'], '0')
        self.assertIn('no change of €0', candidate['answer'])

    def test_native_unit_scale_follows_literal_source_and_never_names_unproved_currency(self):
        candidate = requested_period_candidate(QUESTION, source(unit='$'))
        self.assertIn('a decrease of $40', candidate['answer'])
        self.assertNotIn('HK', candidate['answer'])

    def test_candidate_answer_plan_and_source_identity_tampering_fail_recheck(self):
        catalog = source()
        candidate = requested_period_candidate(QUESTION, catalog)
        self.assertIsNotNone(validate_period_candidate(QUESTION, {'source_period_candidate':candidate}, catalog))
        for mutation in ('answer','plan','source'):
            supplied, changed = copy.deepcopy(candidate), copy.deepcopy(catalog)
            if mutation == 'answer': supplied['answer'] += ' unsupported'
            elif mutation == 'plan': supplied['source_field_plan']['rows'][0]['signed_change'] = '1'
            else: changed['diagnostic-image']['doc_sha256'] = 'c'*64
            with self.assertRaises(ValueError):
                validate_period_candidate(QUESTION, {'source_period_candidate':supplied}, changed)
        self.assertIsNone(validate_period_candidate(QUESTION, {}, catalog))

    def test_no_unregistered_context_or_requested_ratio_shortcut(self):
        with self.assertRaises(ValueError):
            compose_source_period_candidate(QUESTION, {}, object())
        self.assertIsNone(requested_period_candidate(QUESTION.replace('differ','compare in percentage'), source()))


if __name__ == '__main__':
    unittest.main()
