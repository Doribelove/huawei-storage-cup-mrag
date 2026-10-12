import copy
import unittest
from evidence_lab.native_paragraph_quote import (FIELD,requested_native_paragraph_quote,
    validate_native_paragraph_quote,compose_native_paragraph_quote,native_quote_source_declaration)
from evidence_lab.source_prose_quote import quote_claim_groups
from evidence_lab.fact_support import parse_fact_verdict
from evidence_lab.question_focus import condition_focus

QUESTION='How do measurements differ under varying exposure conditions?'
FOCUS=condition_focus(QUESTION)


def fixture():
    texts=['  High exposure gives a complete trace for A but only partial traces for B.\nNo trace is seen for C.  ',
           'Low exposure retains A only.\nNo trace does not prove absence: all compounds were confirmed by another assay.']
    return {f'image-{i}':{'image_index':i,'role':'primary_retrieved_region','source_region_type':'text',
        'doc_sha256':'a'*64,'document_name':'synthetic-exposure-control','pdf_member':'fixture.pdf',
        'page':i,'parent_evidence_id':f'region-{i}','binding_id':f'binding-{i}',
        'image':{'sha256':str(i)*64},'source_text_for_exact_quote':text,
        'local_original_pdf_path':'/synthetic/not-a-real-source.pdf',
        'native_paragraph_original_bbox_top_left_points':[0,0,100,100],
        'readable_viewport':{'native_paragraph_window':True,'native_visible_text':text,
            'native_word_records':[{'text':text,'x0':0,'x1':99,'top':0,'bottom':10}]}}
        for i,text in enumerate(texts,1)}


def validated(candidate):
    return {FIELD:copy.deepcopy(candidate),'diagnostic_candidate_output':{'effective_candidate_output':{
        'answer':candidate['answer'],'citations':[{'image_id':i} for i in candidate['evidence_ids']]}}}


class NativeParagraphQuoteTests(unittest.TestCase):
    def test_whole_native_quotes_keep_qualifiers_and_caveats_without_invented_claims(self):
        catalog=fixture();before=copy.deepcopy(catalog)
        c=requested_native_paragraph_quote(QUESTION,catalog,FOCUS)
        self.assertEqual(c['answer'],'\n\n'.join(s['source_text_for_exact_quote'].strip() for s in catalog.values()))
        for text in ('only partial','No trace','does not prove absence','all compounds'):
            self.assertIn(text,c['answer'])
        self.assertFalse(c['semantic_support_established'])
        self.assertEqual(catalog,before)
        groups=quote_claim_groups(c['answer'],c['source_quote_plan'],validated(c)['diagnostic_candidate_output']['effective_candidate_output']['citations'])
        self.assertEqual([g['scope_source_index'] for g in groups],[0,1])
        self.assertEqual('\n\n'.join(g['claim'] for g in groups),c['answer'])
        for b in c['source_quote_plan']['blocks']:
            self.assertEqual(catalog[b['image_id']]['source_text_for_exact_quote'][slice(*b['source_character_span'])],b['quote'])

    def test_wrong_question_duplicate_geometry_nontext_or_mismatched_native_text_declines(self):
        catalog=fixture()
        self.assertIsNone(requested_native_paragraph_quote('Which objects are shown?',catalog,FOCUS))
        for key,value in (('page',1),('image_index',1),('binding_id','binding-1'),
                          ('parent_evidence_id','region-1'),('doc_sha256','b'*64),
                          ('source_region_type','image'),('source_text_for_exact_quote','Invented figure caption')):
            bad=copy.deepcopy(catalog);bad['image-2'][key]=value
            with self.subTest(key=key):
                self.assertIsNone(requested_native_paragraph_quote(QUESTION,bad,FOCUS))
        bad=copy.deepcopy(catalog);bad['image-2']['readable_viewport']['native_word_records']=[]
        self.assertIsNone(requested_native_paragraph_quote(QUESTION,bad,FOCUS))

    def test_individual_whole_block_over_existing_parser_limit_declines_without_truncation(self):
        catalog=fixture();catalog['image-1']['source_text_for_exact_quote']='a'*4001
        catalog['image-1']['readable_viewport']['native_visible_text']='a'*4001
        self.assertIsNone(requested_native_paragraph_quote(QUESTION,catalog,FOCUS))

    def test_modified_quote_span_binding_or_effective_answer_rejected_before_pdf_read(self):
        catalog=fixture();c=requested_native_paragraph_quote(QUESTION,catalog,FOCUS)
        for key in ('quote','source_character_span','answer_character_span','binding_id','original_image_sha256'):
            bad=validated(c);bad[FIELD]['source_quote_plan']['blocks'][0][key]='altered'
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_native_paragraph_quote(QUESTION,bad,catalog)
        for field,value in (('answer',c['answer'].replace('only partial','complete')),
                            ('citations',[{'image_id':'image-1'}])):
            bad=validated(c);bad['diagnostic_candidate_output']['effective_candidate_output'][field]=value
            with self.assertRaises(ValueError):validate_native_paragraph_quote(QUESTION,bad,catalog)

    def test_declared_source_retains_identity_and_requires_its_actual_claim_scope(self):
        import json
        catalog=fixture();c=requested_native_paragraph_quote(QUESTION,catalog,FOCUS)
        effective=validated(c)['diagnostic_candidate_output']['effective_candidate_output']
        groups=quote_claim_groups(c['answer'],c['source_quote_plan'],effective['citations'])
        declared=native_quote_source_declaration(0,effective['citations'][0],catalog['image-1'],c['source_quote_plan'])
        self.assertTrue(declared['actual_complete_original_source_pixels_follow'])
        self.assertFalse(declared['exact_native_word_binding_is_answer_semantic_proof'])
        verdict={'claims':[{'id':i,'supported':True,'evidence':[1-i],'reason':'wrong source fixture'} for i in range(2)],
                 'question_answered':True,'reason':'fixture'}
        with self.assertRaises(ValueError):parse_fact_verdict(json.dumps(verdict),groups,2)

    def test_no_acquisition_has_no_effect_and_unregistered_context_cannot_compose(self):
        generated={'raw_output_for_validation':'original diagnostic bytes'}
        self.assertIs(compose_native_paragraph_quote(QUESTION,generated,object(),{}),generated)
        with self.assertRaises(ValueError):
            compose_native_paragraph_quote(QUESTION,generated,object(),{'candidate_applied':True})
        self.assertIsNone(validate_native_paragraph_quote(QUESTION,{},{}))


if __name__=='__main__':unittest.main()
