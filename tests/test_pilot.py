"""Pilot sampling/context fixtures; no model or QA labels are used as evidence."""
import unittest
import json
from pathlib import Path
from evidence_lab.safe_data import sha256_file
from evidence_lab.pilot import ContextAssembler,answer_support_state,select_pilot


def fixture(eid,order,page=1,name='unit',pdf='pdf',text='Heading',member='parsed.json'):
    return {'role':'source_pdf_region','evidence_id':eid,'doc_sha256':pdf,'document_name':name,
        'page':page,'type':'text','text':text,'source_bbox':[0,0,100,100],
        'aliases':[{'document_name':name,'content_list_member':member,'record_index':order}]}


class PilotTests(unittest.TestCase):
    def test_frozen_manifest_bytes_match_diagnostic_binding(self):
        configs=Path(__file__).resolve().parents[1]/'configs'
        diagnostics=json.loads((configs/'pilot-v1.diagnostics.json').read_text(encoding='utf-8'))
        self.assertEqual(sha256_file(configs/'pilot-v1.manifest.json'),diagnostics['manifest_sha256'])
        self.assertEqual(len(diagnostics['cases']),14)

    def test_sampler_is_deterministic_order_independent_and_source_stratified(self):
        docs={}
        for domain in 'abcdefgh':
            for number,pages in enumerate([5,15,25]):
                name=f'{domain}_{number}';docs[name]={'sha256':name,'parsed_body_sha256':name,'page_count':pages}
        review=[{'document_name':'a_2','visual_classification':'parse-missing','anomalies':[{'type':'table'}]},
                {'document_name':'b_2','visual_classification':'parse-missing','anomalies':[{'type':'table'}]}]
        selected,rule=select_pilot(docs,review)
        reversed_result=select_pilot(dict(reversed(list(docs.items()))),list(reversed(review)))
        self.assertEqual((selected,rule),reversed_result)
        self.assertEqual(len(selected),10);self.assertEqual(len({d['name'] for d in selected}),10)
        self.assertEqual({d['domain'] for d in selected},set('abcdefgh'))
        self.assertEqual({d['page_bin'] for d in selected},{'short','medium','long'})
        self.assertFalse(rule['query_or_answer_inputs'])
        docs['a_0']['gt_answer']='forbidden'
        with self.assertRaises(ValueError): select_pilot(docs,review)

    def test_context_uses_original_order_within_pdf_page_and_member(self):
        rows=[fixture('a',2),fixture('later',4,text='Explanation'),fixture('first',3,text='Paragraph'),
            fixture('otherpage',3,page=2),fixture('otherpdf',3,pdf='different'),fixture('otherdoc',3,name='different'),
            fixture('otherparser',3,member='different.json')]
        assembled=ContextAssembler(list(reversed(rows))).assemble([rows[0]])
        self.assertEqual([r['evidence_id'] for r in assembled['added_regions']],['first','later'])
        self.assertEqual([l['source_record_index'] for l in assembled['links']],[3,4])
        self.assertTrue(all(r['page']==1 and r['doc_sha256']=='pdf' and r['document_name']=='unit' for r in assembled['added_regions']))

    def test_context_budget_and_unambiguous_source_required(self):
        rows=[fixture('a',0),fixture('b',1,text='body'*50),fixture('c',2,text='body'*50)]
        assembled=ContextAssembler(rows).assemble([rows[0]],max_added=1)
        self.assertEqual(len(assembled['added_regions']),1)
        self.assertIn('added_region_budget',assembled['stop_reasons'])
        self.assertEqual(ContextAssembler(rows).assemble([rows[0]],max_characters=10)['added_regions'],[])
        with self.assertRaises(ValueError): ContextAssembler(rows).assemble([{**rows[0],'doc_sha256':'wrong'}])
        ambiguous={**rows[0],'aliases':rows[0]['aliases']*2}
        self.assertEqual(ContextAssembler([ambiguous,*rows[1:]]).assemble([ambiguous])['added_regions'],[])
        with self.assertRaises(ValueError): ContextAssembler([{**rows[0],'gt_answer':'gold'}])

    def test_candidates_never_imply_supported_answer_without_reader(self):
        hit=fixture('a',0)
        states=[answer_support_state([],[]),answer_support_state([hit],[]),
            answer_support_state([hit],[{'evidence_id':'a','all_passed':True}])]
        self.assertEqual([s['candidate_state'] for s in states],['no_candidates','candidate_provenance_failed','candidates_only'])
        self.assertTrue(all(not s['supported_answer'] and not s['answer_generated'] and s['support_state']=='not_established' for s in states))
        self.assertTrue(all(not s['similarity_threshold_calibrated'] for s in states))


if __name__=='__main__': unittest.main()
