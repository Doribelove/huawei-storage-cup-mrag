import copy
import json
import unittest
from evidence_lab.diagram_candidate import (requested_role,parse_diagram,construct_candidate,
    diagram_claim_groups,validate_diagram_candidate,extract_diagram_candidate)
from evidence_lab.fact_support import parse_fact_verdict
from evidence_lab.diagram_window import diagram_window,DISPLAY_QUESTION

def source():
    return {'image_id':'image-fixture','parent_evidence_id':'region-fixture','doc_sha256':'a'*64,
        'document_name':'synthetic-flow','page':1,'binding_id':'b'*64,
        'image':{'size_pixels':[400,300],'sha256':'c'*64},
        'image_pixel_to_pdf_matrix':[.5,0,0,-.5,0,150]}

def diagram():
    return {'caption':'Figure 7. Test workflow','nodes':[
        {'label':'Test workflow','role':'caption','box':[0,900,900,990]},
        {'label':'Start ≥ 2 ms','role':'step','box':[100,100,350,170]},
        {'label':'Recheck if incomplete','role':'step','box':[300,500,700,600]},
        {'label':'Cycle 2','role':'context','box':[0,500,200,550]},
        {'label':'Start ≥ 2 ms','role':'step','box':[100,700,350,770]}],
        'edges':[[1,2],[2,4]]}

class DiagramCandidateTests(unittest.TestCase):
    def test_literal_nodes_preserve_revision_qualifier_operator_and_ignore_caption_context(self):
        original=diagram();before=copy.deepcopy(original)
        plan=construct_candidate('Which stages are in the depicted process?',original,source(),{})
        self.assertEqual(plan['answer'],'Start ≥ 2 ms\nRecheck if incomplete')
        self.assertEqual(original,before)
        self.assertFalse(plan['model_nodes_are_source_quotes'])
        self.assertFalse(plan['semantic_support_established'])
        self.assertEqual(len(plan['model_diagram']['nodes']),5)
        effective={'answer':plan['answer']}
        claims=diagram_claim_groups(effective,plan,[{'image_id':'image-fixture'}])
        self.assertEqual([c['claim'] for c in claims],['Start ≥ 2 ms','Recheck if incomplete'])
        self.assertTrue(all(c['scope_source_index']==0 for c in claims))
        effective['answer']=effective['answer'].replace('≥','>')
        with self.assertRaises(ValueError):diagram_claim_groups(effective,plan,[{'image_id':'image-fixture'}])

    def test_scope_evidence_cannot_be_replaced_by_another_source(self):
        plan=construct_candidate('Which steps are depicted?',diagram(),source(),{})
        claims=diagram_claim_groups({'answer':plan['answer']},plan,[{'image_id':'image-fixture'}])
        verdict={'claims':[{'id':i,'supported':True,'evidence':[1],'reason':'wrong source'} for i in range(2)],
            'question_answered':True,'reason':'fixture'}
        with self.assertRaises(ValueError):parse_fact_verdict(json.dumps(verdict),claims,2)
        for row in verdict['claims']:row['evidence']=[0];row['supported']=False
        self.assertFalse(parse_fact_verdict(json.dumps(verdict),claims,2)[2])

    def test_malformed_observation_never_repaired_into_a_candidate(self):
        for change in ('role','box','edge','duplicate'):
            bad=diagram()
            if change=='role':bad['nodes'][1]['role']='verified_answer'
            elif change=='box':bad['nodes'][1]['box']=[-1,0,200,100]
            elif change=='edge':bad['edges']=[[1,8]]
            else:bad['nodes'].append(copy.deepcopy(bad['nodes'][1]))
            with self.subTest(change=change),self.assertRaises(ValueError):parse_diagram(json.dumps(bad))
        with self.assertRaises(ValueError):parse_diagram('{"caption":"","caption":"","nodes":[],"edges":[]}')
        self.assertIsNone(construct_candidate('Which weaknesses are shown?',diagram(),source(),{}))
        self.assertIsNone(requested_role('Which phases and weaknesses are depicted?'))
        self.assertIsNone(requested_role('What is the pressure?'))

    def test_no_label_from_a_caption_alone_and_no_fake_coordinate_repair(self):
        d=diagram();d['nodes']=[d['nodes'][0]];d['edges']=[]
        self.assertIsNone(construct_candidate('Which steps are shown?',d,source(),{}))
        row={'type':'image','bbox_points_estimate':[20,20,180,150],
            'bbox_coordinate_system':'MinerU integer 0..1000, top-left; PDF points estimate is quantized'}
        words=[{'text':'Caption','x0':20,'top':160,'x1':180,'bottom':170},
               {'text':'outside','x0':-5,'top':30,'x1':50,'bottom':40}]
        plan=diagram_window(DISPLAY_QUESTION,row,words,200,220)
        self.assertFalse(plan['source_caption_is_answer_sentence'])
        self.assertFalse(plan['fact_localization_verified'])
        self.assertEqual(plan['native_page_word_geometry_audit']['excluded_nonvisible_word_count'],1)
        self.assertEqual(plan['native_object_bbox_top_left_points'],[20,20,180,150])
        with self.assertRaises(ValueError):diagram_window('a task-specific directive',row,words,200,220)

    def test_shared_extra_call_budget_has_no_effect_on_original_candidate(self):
        original={'answer_candidate_present':True};context=object()
        output=extract_diagram_candidate('Which steps are shown?',original,context,
            selected=[],visual=None,search=object(),reader=None,processor=None,budget=None,
            config={},output=None,prior_calls=1)
        self.assertIs(output[0],original);self.assertIs(output[1],context)
        self.assertFalse(output[2]['eligible'])
        self.assertEqual(output[2]['attempted_transcription_calls'],0)
        self.assertIsNone(validate_diagram_candidate('test',{},{}))

if __name__=='__main__':unittest.main()
