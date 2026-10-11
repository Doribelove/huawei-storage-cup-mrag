import copy
import unittest
from evidence_lab.native_fields import aligned_header, parameter_rows, comparison_records, apply_comparisons, grounded_cell_rows
from evidence_lab.source_semantics import deterministic_audit, financial_change_plan, decimal_value, caption_context_conflicts


def w(text,x,y,width=12):
    return {'text':text,'x0':x,'x1':x+width,'top':y,'bottom':y+10}


class NativeFieldsTests(unittest.TestCase):
    def table(self):
        return [w('Table',10,20),w('A',94,50),w('B',194,50),w('C',294,50),
                w('1.2',94,90),w('2.3',194,90),w('3.4',294,90),
                w('ω',50,96,6),w('2',56,100,4),
                w('4.5',94,130),w('5.6',194,130),w('6.7',294,130),
                w('β',50,136,6),w('2',56,140,4)]

    def test_header_recovery_retains_title_labels_and_coordinates(self):
        result=aligned_header(self.table(),[0,85,400,155])
        self.assertEqual(result['header_labels'],['A','B','C'])
        self.assertEqual(result['title'],'Table')
        self.assertEqual(result['top'],2)

    def test_missing_or_misaligned_header_is_not_invented(self):
        words=[x for x in self.table() if x['top']!=50]
        self.assertIsNone(aligned_header(words,[0,85,400,155]))
        words=self.table();words[2]['x0']=165;words[2]['x1']=177
        self.assertIsNone(aligned_header(words,[0,85,400,155]))

    def test_intervening_table_title_blocks_header_inheritance(self):
        words=self.table()+[w('Table',10,75)]
        self.assertIsNone(aligned_header(words,[0,85,400,155]))

    def test_subscript_label_is_bound_to_estimate_not_interval(self):
        rows=parameter_rows(self.table())
        self.assertEqual([r['row_label'] for r in rows],['ω2','β2'])
        self.assertEqual(rows[1]['values'],['4.5','5.6','6.7'])
        self.assertEqual(len(rows[0]['value_boxes']),3)

    def test_parameter_wrong_row_rejected_and_true_row_preserved(self):
        catalog={'x':{'readable_viewport':{'native_parameter_rows':parameter_rows(self.table())}}}
        self.assertFalse(deterministic_audit('Compare ω2 and β2','For ω2, the value is 5.6.',catalog)['passed'])
        audit=deterministic_audit('Compare ω2 and β2','For ω2, the value is 2.3; β2 is 5.6.',catalog)
        self.assertTrue(audit['passed']);self.assertFalse(audit['establishes_support'])
        self.assertFalse(deterministic_audit('Compare parameters','For ω₂, the value is 5.6.',catalog)['passed'])

    def test_short_vector_underline_restores_inclusive_operator(self):
        char=w('>',50,100,6)
        edge={'x0':50,'x1':58,'top':108.5,'bottom':108.5}
        records=comparison_records([char],[edge],[0,0,400,400])
        self.assertEqual(records[0]['operator'],'≥')
        original=[char];before=copy.deepcopy(original)
        self.assertEqual(apply_comparisons(original,records)[0]['text'],'≥')
        self.assertEqual(original,before)

    def test_table_border_and_distant_underline_do_not_change_strict_operator(self):
        char=w('>',50,100,6)
        for edge in ({'x0':10,'x1':300,'top':110,'bottom':110},
                     {'x0':50,'x1':58,'top':115,'bottom':115}):
            self.assertEqual(comparison_records([char],[edge],[0,0,400,400])[0]['operator'],'>')

    def test_inclusive_boundary_mismatch_rejected_with_same_row_and_unit(self):
        record={'row_label':'Example therapy','source_line':'Pre-treatment: ≥ 4 weeks','operator':'≥'}
        catalog={'x':{'readable_viewport':{'native_comparison_records':[record]}}}
        self.assertFalse(deterministic_audit('Timing for Example therapy?','>4 weeks',catalog)['passed'])
        self.assertTrue(deterministic_audit('Timing for Example therapy?','at least 4 weeks',catalog)['passed'])
        self.assertTrue(deterministic_audit('Timing for Example therapy?','AT LEAST 4 WEEKS',catalog)['passed'])
        self.assertFalse(deterministic_audit('Timing for Example therapy?','≤4 weeks',catalog)['passed'])
        self.assertFalse(deterministic_audit('Timing for Other therapy?','>4 weeks',catalog)['applicable'])

    def test_native_cell_boundary_correction_preserves_raw_cells_and_other_operator(self):
        cells=[['Therapy','Pre: > 4 weeks\nPost: > 6 months']];before=copy.deepcopy(cells)
        record={'table_index':0,'row_index':0,'column_index':1,'extracted_glyph':'>',
                'operator':'≥','source_line':'Pre: ≥ 4 weeks'}
        value=grounded_cell_rows(cells,[record],0)
        self.assertEqual(value[0][1],'Pre: ≥ 4 weeks\nPost: > 6 months');self.assertEqual(cells,before)
        self.assertEqual(grounded_cell_rows(cells,[record],1),cells)

    def test_boundary_row_binding_never_borrows_operator_from_other_sentence(self):
        catalog={'x':{'readable_viewport':{'native_comparison_records':[
            {'row_label':'First therapy','source_line':'First therapy ≥4 weeks'},
            {'row_label':'Second therapy','source_line':'Second therapy >4 weeks'}]}}}
        answer='First therapy: ≥4 weeks. Second therapy: >4 weeks.'
        audit=deterministic_audit('Timing for First therapy?',answer,catalog)
        self.assertTrue(audit['passed']);self.assertEqual(len(audit['checks']),2)
        self.assertEqual(audit['checks'][0]['candidate_clauses'],['First therapy: ≥4 weeks.'])
        self.assertFalse(audit['establishes_support'])
        self.assertFalse(deterministic_audit('Timing for First therapy?',
            'First therapy: >4 weeks. Second therapy: ≥4 weeks.',catalog)['passed'])

    def test_ambiguous_mixed_row_clause_is_not_promoted_to_operator_binding(self):
        catalog={'x':{'readable_viewport':{'native_comparison_records':[
            {'row_label':'First therapy','source_line':'First therapy ≥4 weeks'},
            {'row_label':'Second therapy','source_line':'Second therapy >4 weeks'}]}}}
        audit=deterministic_audit('Timing for First therapy?',
            'First therapy ≥4 weeks and Second therapy >4 weeks.',catalog)
        self.assertFalse(audit['applicable']);self.assertFalse(audit['establishes_support'])

    def test_ambiguous_repeated_cell_threshold_is_not_changed(self):
        cells=[['Therapy','First: > 4 weeks\nSecond: > 4 weeks']]
        record={'table_index':0,'row_index':0,'column_index':1,'extracted_glyph':'>',
                'operator':'≥','source_line':'First: ≥ 4 weeks'}
        self.assertEqual(grounded_cell_rows(cells,[record],0),cells)

    def financial(self):
        words=[w('Share',145,30),w('capital',145,42),w('Retained',245,30),w('earnings',245,42),
               w('€',145,54),w('€',245,54)]
        for y,year,value in [(80,'2021','(1,200.25)'),(110,'2022','(1,340.75)')]:
            x=10
            for text in ['Balance','at','31','December',year]:
                words.append(w(text,x,y,20));x+=23
            words.extend([w('500',145,y),w(value,245,y)])
        return {'x':{'image':{'sha256':'source-hash'},'readable_viewport':{'native_word_records':words}}}

    def test_decimal_change_uses_named_component_with_source_coordinates(self):
        plan=financial_change_plan('Change in retained earnings from 2021 to 2022?',self.financial())
        self.assertEqual(plan['signed_change'],'-140.50')
        self.assertEqual(plan['unit'],'€');self.assertTrue(plan['operand_coordinates']['start'])
        self.assertEqual(decimal_value('(1,200.25)'),decimal_value('-1200.25'))

    def test_financial_wrong_total_wrong_unit_and_extra_operand_rejected(self):
        q='Change in retained earnings from 2021 to 2022?';c=self.financial()
        for answer in ['500','-140.50 USD','-140.50, from 12 to 152.5']:
            self.assertFalse(deterministic_audit(q,answer,c)['passed'])
        self.assertTrue(deterministic_audit(q,'decreased by 140.50 euros',c)['passed'])

    def test_financial_ambiguous_period_abstains(self):
        catalog=self.financial();catalog['x']['readable_viewport']['native_word_records'] += [
            w('Balance',10,140,20),w('at',33,140,20),w('1',56,140,20),w('July',79,140,20),
            w('2021',102,140,20),w('500',145,140),w('(900)',245,140)]
        self.assertIsNone(financial_change_plan('Change in retained earnings from 2021 to 2022?',catalog))

    def test_correct_list_without_caption_is_not_subject_to_generated_inventory(self):
        audit=deterministic_audit('List the process steps','Choose product; Draw present; Draw future; Implement',{})
        self.assertTrue(audit['passed']);self.assertFalse(audit['applicable'])

    def test_disaster_context_does_not_prove_material_performance(self):
        text='Earthquake survivors require reconstruction. This creates demand for building materials. Exports increased. '
        catalog={'x':{'source_region_type':'text','readable_viewport':{'native_visible_text':text}}}
        audit=deterministic_audit('Why demand?','Interest includes earthquake-safe building needs.',catalog)
        self.assertFalse(audit['passed']);self.assertFalse(audit['establishes_support'])

    def test_explicit_performance_and_seismic_synonym_remain_eligible(self):
        text='Testing established seismic resistance in structural panels. '+('Context. '*10)
        catalog={'x':{'source_region_type':'text','readable_viewport':{'native_visible_text':text}}}
        self.assertTrue(deterministic_audit('Why demand?','Earthquake-resistant panels.',catalog)['passed'])

    def test_visual_performance_is_not_judged_by_native_text_absence(self):
        catalog={'x':{'source_region_type':'image','readable_viewport':{'native_visible_text':'Caption. '*20}}}
        self.assertFalse(deterministic_audit('Why demand?','Fire-resistant panel.',catalog)['applicable'])

    def test_caption_scene_context_cannot_certify_visual_machine_identity(self):
        verdict={'claims':[{'id':0,'supported':True,'reason':"Image caption 'Street maintenance' matches claim context."}]}
        conflicts=caption_context_conflicts(verdict,'Which machinery is depicted?')
        self.assertEqual(conflicts[0]['type'],'caption_context_without_requested_visual_identity')
        self.assertTrue(verdict['claims'][0]['supported'])

    def test_actual_object_reason_and_requested_caption_context_remain_eligible(self):
        verdict={'claims':[{'id':0,'supported':True,'reason':'Image depicts a loader beside a truck.'}]}
        self.assertEqual(caption_context_conflicts(verdict,'Which machinery is depicted?'),[])
        verdict['claims'][0]['reason']="Image caption 'River Street' matches claim context."
        self.assertEqual(caption_context_conflicts(verdict,'Which street is the scene on?'),[])
