import unittest
from evidence_lab.readable_region import choose_window,native_word_lines
from evidence_lab.bounded_selection import select_question_pages
from evidence_lab.reader_context import native_window_projection


class ReadableRegionTests(unittest.TestCase):
    def test_native_payload_keeps_complete_rows_structured_and_unverified(self):
        import copy,json
        viewport={'native_spatial_word_lines':[['Label','2024','2023'],['Asset','10.001','20.002']],
                  'source_cell_rows':None}
        before=copy.deepcopy(viewport)
        projection=native_window_projection(viewport,'original record retained elsewhere')
        payload=json.loads(json.dumps({'text':projection}))
        self.assertIsInstance(payload['text'],dict)
        self.assertEqual(payload['text']['NATIVE_SPATIAL_WORD_LINES'],viewport['native_spatial_word_lines'])
        self.assertFalse(payload['text']['spatial_word_groups_are_semantic_cells'])
        self.assertEqual(viewport,before)
        self.assertEqual(native_window_projection({},'exact original'),'exact original')

    def test_native_word_lines_preserve_values_and_horizontal_groups(self):
        words=[{'text':'State','top':20,'x0':10,'x1':30},
               {'text':'2','top':20,'x0':34,'x1':40},
               {'text':'CC','top':20,'x0':80,'x1':95},
               {'text':'η2','top':40,'x0':10,'x1':25},
               {'text':'4.165','top':40,'x0':80,'x1':110}]
        self.assertEqual(native_word_lines(words),[['State 2','CC'],['η2','4.165']])
        self.assertEqual(words[-1]['text'],'4.165')

    def test_actual_table_box_and_caption_margin_derive_bounded_high_resolution_window(self):
        items=[{'kind':'native_table','index':0,'bbox':[20,100,500,300],
                'text':'Current tax 2024 2023 Hong Kong','cells':[['2024','2023'],['6','12']]}]
        result=choose_window('How did current tax change in 2024?',{'type':'table','text':'tax'},items,600,800)
        self.assertEqual(result['viewport_bbox_top_left_points'],[0,72,600,328])
        self.assertEqual(result['dpi'],144)
        self.assertEqual(result['source_cell_rows'],items[0]['cells'])
        self.assertFalse(result['association_verified'])

    def test_numeric_only_box_retains_row_labels_outside_its_horizontal_extent(self):
        result=choose_window('Compare current liabilities',{'type':'table'},[
            {'kind':'source_region_native_word_union','index':0,'bbox':[400,150,520,240],
             'text':'2024 2023 10 20','cells':None}],600,800)
        self.assertEqual(result['viewport_bbox_top_left_points'],[0,122,600,268])
        self.assertTrue(result['full_width_row_label_context_retained'])
        self.assertFalse(result['association_verified'])

    def test_no_object_and_text_page_use_explicit_full_page_fallback(self):
        result=choose_window('What happened?',{'type':'text','text':'x'},[],600,800)
        self.assertEqual(result['status'],'full_page_fallback')

    def test_visual_caption_match_alone_does_not_replace_text_page(self):
        item={'kind':'embedded_pdf_image','index':0,'bbox':[100,100,300,400],
              'text':'Cyclical research stages and revision'}
        result=choose_window('Which stages occur in cyclical research?',{'type':'text'},[item],600,800)
        self.assertEqual(result['status'],'full_page_fallback')
        self.assertFalse(result['association_verified'])

    def test_unrelated_image_caption_cannot_replace_text_source(self):
        item={'kind':'embedded_pdf_image','index':0,'bbox':[100,100,300,400],'text':'Company logo'}
        result=choose_window('Which stages occur in cyclical research?',{'type':'text'},[item],600,800)
        self.assertEqual(result['status'],'full_page_fallback')

    def test_performance_cue_triggers_bounded_source_gap_without_asserting_safety(self):
        from evidence_lab.bounded_selection import required_terms
        self.assertIn('safe',required_terms('Earthquake-safe buildings?'))
        hits=[{'evidence_id':'A','doc_sha256':'X','page':1,'text':'Earthquake reconstruction demand.'}]
        _,report=select_question_pages('Why earthquake-safe buildings?',hits)
        self.assertIn('safe',report['missing_exact_source_search_cues'])
        self.assertTrue(report['possible_missing_fields']);self.assertTrue(report['heuristic_not_semantic_coverage'])

    def test_unruled_native_word_window_does_not_invent_cells(self):
        result=choose_window('Which state has the largest rate?',{'type':'table'},[
            {'kind':'source_region_native_word_union','index':0,'bbox':[30,150,500,400],
             'text':'state rate','cells':None}],600,800)
        self.assertEqual(result['status'],'selected_native_pdf_window')
        self.assertIsNone(result['source_cell_rows'])
        self.assertFalse(result['table_continuity_verified'])
        result=choose_window('What happened?',{'type':'text','text':'x'},[
            {'kind':'embedded_pdf_image','index':0,'bbox':[10,10,300,300],'text':'logo'}],600,800)
        self.assertEqual(result['status'],'full_page_fallback')

    def test_invalid_or_invented_geometry_is_not_clamped_into_source(self):
        for box in ([-1,0,100,100],[10,10,5,20],[0,0,700,100]):
            with self.assertRaises(ValueError):choose_window('Which table?',{'type':'table'},[
                {'kind':'native_table','index':0,'bbox':box,'text':'tax'}],600,800)

    def test_two_page_selection_keeps_primary_and_marks_missing_year_without_using_labels(self):
        def row(eid,doc,page,text):return {'evidence_id':eid,'doc_sha256':doc,'page':page,'text':text}
        hits=[row('A','X',1,'Taxes 2024'),row('B','Y',1,'Unrelated school'),row('C','X',2,'Taxes 2023')]
        selected,report=select_question_pages('How did taxes change from 2023 to 2024?',hits)
        self.assertEqual([r['evidence_id'] for r in selected],['A','C'])
        self.assertFalse(report['possible_missing_fields'])
        self.assertEqual(hits[1]['evidence_id'],'B')
        selected,report=select_question_pages('Taxes 2020?',hits)
        self.assertEqual(report['missing_exact_source_search_cues'],['2020'])


if __name__=='__main__':unittest.main()
