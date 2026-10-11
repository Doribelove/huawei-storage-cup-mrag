import copy,unittest
from decimal import Decimal
from evidence_lab.confusion_fields import native_matrix,confusion_rate_plan,rate_checks
from evidence_lab.source_semantics import compact_derivation,deterministic_audit

def word(text,x,y,width=10):return {'text':text,'x0':x,'x1':x+width,'top':y,'bottom':y+9}

def fixture(columns=('A','B','C'),reverse_axes=False):
    labels=['A','B','C'];values=[[42,3,5],[1,17,2],[4,2,24]]
    words=[word('Table',0,100,22),word('9.',24,100),word('Confusion',40,100,48),word('matrix',90,100,30),
        word('Prediction' if reverse_axes else 'Reference',140,110,40)]
    words.extend(word(label,100+j*60,130) for j,label in enumerate(columns))
    words.append(word('Reference' if reverse_axes else 'Prediction',5,170,30))
    for i,label in enumerate(labels):
        words.append(word(label,40,150+i*20))
        words.extend(word(str(values[i][labels.index(label2)]),100+j*60,150+i*20)
            for j,label2 in enumerate(columns))
    return {'native_word_records':words,'native_paint_filter':{'schema':'native-form-bbox-paint-filter-v1'}}

def catalog(viewport):return {'image-test':{'role':'primary_retrieved_region','doc_sha256':'d'*64,
    'page':3,'image':{'sha256':'a'*64},'readable_viewport':viewport}}

class ConfusionFieldsTests(unittest.TestCase):
    def test_body_mention_is_not_a_second_table_title(self):
        viewport=fixture()
        viewport['native_word_records'].append(word('Theconfusionmatrixisreportedbelow',0,85,190))
        self.assertIsNotNone(native_matrix(viewport))

    def test_prose_mention_cannot_replace_missing_caption(self):
        viewport=fixture()
        viewport['native_word_records']=[w for w in viewport['native_word_records'] if w['top']!=100]
        viewport['native_word_records'].append(word('Theconfusionmatrixisreportedbelow',0,85,190))
        self.assertIsNone(native_matrix(viewport))

    def test_plain_caption_both_axis_denominators_and_cell_coordinates(self):
        viewport=fixture();before=copy.deepcopy(viewport)
        plan=confusion_rate_plan('Compare misclassification rates by class',catalog(viewport))
        self.assertIsNotNone(plan);self.assertEqual(plan['title'],'Table 9. Confusion matrix')
        a=plan['row_conditioned_rates'][0];b=plan['column_conditioned_rates'][0]
        self.assertEqual((a['fraction'],a['percent_rounded_2dp']),('8/50','16.00'))
        self.assertEqual((b['fraction'],b['percent_rounded_2dp']),('5/47','10.64'))
        self.assertEqual(len(plan['source_coordinates']['cells']),3)
        self.assertEqual(viewport,before);self.assertFalse(plan['establishes_support'])
        self.assertNotIn('source_coordinates',compact_derivation(plan))

    def test_diagonal_uses_class_label_when_column_order_is_changed(self):
        plan=confusion_rate_plan('misclassification rates',catalog(fixture(('B','A','C'))))
        self.assertEqual(plan['column_labels'],['B','A','C'])
        self.assertEqual([r['percent_rounded_2dp'] for r in plan['row_conditioned_rates']],['16.00','15.00','20.00'])

    def test_reversed_axes_keep_actual_conditioning_labels(self):
        plan=confusion_rate_plan('misclassification rates',catalog(fixture(reverse_axes=True)))
        self.assertEqual(plan['row_conditioned_rates'][0]['conditioning_axis'],'reference')
        self.assertEqual(plan['column_conditioned_rates'][0]['conditioning_axis'],'prediction')

    def test_ambiguous_missing_or_misaligned_fields_abstain(self):
        for mutate in ('axis','title','alignment','duplicate','percentage','paint_metadata'):
            v=fixture()
            if mutate=='axis':v['native_word_records']=[w for w in v['native_word_records'] if w['text']!='Reference']
            elif mutate=='title':v['native_word_records']=[w for w in v['native_word_records'] if w['top']!=100]
            elif mutate=='alignment':v['native_word_records'][-1]['x0']+=20;v['native_word_records'][-1]['x1']+=20
            elif mutate=='duplicate':v['native_word_records'].append(dict(v['native_word_records'][4]))
            elif mutate=='percentage':v['native_word_records'].append(word('percentages',125,100,60))
            else:v.pop('native_paint_filter')
            self.assertIsNone(native_matrix(v),mutate)

    def test_wrong_percentages_and_wrong_explicit_denominators_reject(self):
        c=catalog(fixture());plan=confusion_rate_plan('misclassification rates',c)
        correct='Predicted misclassification rates: A 16%, B 15%, C 20%.'
        checks=rate_checks(plan,correct);self.assertEqual(len(checks),3);self.assertTrue(all(x['passed'] for x in checks))
        bad=rate_checks(plan,'Predicted misclassification rates: A 10.64%, B 15%, C 20%.')
        self.assertFalse(bad[0]['passed'])
        self.assertFalse(deterministic_audit('misclassification rates','Misclassification rates for A, B, C are 7%, 2%, 9%, respectively.',c)['passed'])
        audit=deterministic_audit('misclassification rates',correct,c)
        self.assertTrue(audit['passed']);self.assertFalse(audit['establishes_support'])
        self.assertEqual(rate_checks(plan,'The source image shows three categories.'),[])

    def test_nonprimary_or_other_metric_does_not_supply_arithmetic(self):
        c=catalog(fixture());c['image-test']['role']='candidate_context'
        self.assertIsNone(confusion_rate_plan('misclassification rates',c))
        self.assertIsNone(confusion_rate_plan('What is the energy demand?',catalog(fixture())))

    def test_literal_off_diagonal_directions_follow_axes_not_row_index(self):
        p=confusion_rate_plan('misclassification rates',catalog(fixture()))
        found={(r['reference_class'],r['predicted_class']):r['literal_matrix_value']
            for r in p['source_observed_off_diagonal_cells']}
        self.assertEqual(found[('A','B')],'1');self.assertEqual(found[('B','A')],'3')
        reverse=confusion_rate_plan('misclassification rates',catalog(fixture(reverse_axes=True)))
        actual={(r['reference_class'],r['predicted_class']):r['literal_matrix_value']
            for r in reverse['source_observed_off_diagonal_cells']}
        self.assertEqual(actual[('A','B')],'3');self.assertEqual(actual[('B','A')],'1')

    def test_two_suffix_scopes_check_every_percentage_and_reject_each_bad_view(self):
        p=confusion_rate_plan('misclassification rates',catalog(fixture()))
        text='Misclassification rates for A, B, C are 16%, 15%, 20% respectively when conditioned on prediction, and 10.64%, 22.73%, 22.58% respectively when conditioned on reference.'
        checks=rate_checks(p,text)
        self.assertEqual(len(checks),6);self.assertTrue(all(c['passed'] for c in checks))
        for bad in (text.replace('16%','18%'),text.replace('10.64%','16%')):
            changed=rate_checks(p,bad);self.assertEqual(len(changed),6)
            self.assertFalse(all(c['passed'] for c in changed))
        self.assertEqual(rate_checks(p,'Prediction accuracy rates for A, B, C are 84%, 85%, 80%, respectively.'),[])

    def test_two_prefix_scopes_preserve_correct_views_and_reject_each_bad_view(self):
        p=confusion_rate_plan('misclassification rates',catalog(fixture()))
        text='Misclassification rates for A, B, C are when conditioned on prediction 16%, 15%, 20%, respectively, and when conditioned on reference 10.64%, 22.73%, 22.58%, respectively.'
        checks=rate_checks(p,text)
        self.assertEqual(len(checks),6);self.assertTrue(all(c['passed'] for c in checks))
        for bad in (text.replace('16%','18%'),text.replace('10.64%','16%')):
            changed=rate_checks(p,bad);self.assertEqual(len(changed),6)
            self.assertFalse(all(c['passed'] for c in changed))
