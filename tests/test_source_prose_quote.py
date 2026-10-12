"""Synthetic scope/integrity controls, never benchmark answers or semantic scores."""
import copy
import unittest
from evidence_lab.bounded_selection import qualification_gap_request
from evidence_lab.fact_support import parse_fact_verdict
from evidence_lab.source_prose_quote import (FIELD, requested_qualification_quote_candidate,
    validate_qualification_quote_candidate, quote_claim_groups, compose_qualification_quote_candidate)


QUESTION='Why is ABC of interest for fire-resistant buildings and exports, and how is ABC used?'


def fixture():
    texts=['  ABC has limited domestic use; most output is exported. Only a few houses use ABC.  ',
           'Recent fires motivate promotion of ABC. Actual domestic use is not widespread.',
           'ABC is fire resistant under the stated dry test conditions. Wet exposure was not tested.']
    catalog={f'image-{i}':{'image_index':i,'role':'primary_retrieved_region','source_region_type':'text',
        'doc_sha256':'a'*64,'document_name':'synthetic-scope-control','pdf_member':'fixture.pdf',
        'page':i,'parent_evidence_id':f'region-{i}','binding_id':f'binding-{i}',
        'image':{'sha256':str(i)*64},'source_text_for_exact_quote':text}
        for i,text in enumerate(texts,1)}
    request=qualification_gap_request(QUESTION,[{'doc_sha256':'a'*64,'text':t} for t in texts[:2]])
    return catalog,request


def validated(candidate):
    return {FIELD:copy.deepcopy(candidate),'diagnostic_candidate_output':{'effective_candidate_output':{
        'answer':candidate['answer'],'citations':[{'image_id':i} for i in candidate['evidence_ids']]}}}


class SourceProseQuoteTests(unittest.TestCase):
    def test_all_original_scope_negations_conditions_and_bytes_are_retained(self):
        catalog,request=fixture();before=copy.deepcopy(catalog)
        c=requested_qualification_quote_candidate(QUESTION,catalog,'region-3',request)
        self.assertEqual(c['answer'],'\n\n'.join(x['source_text_for_exact_quote'].strip() for x in catalog.values()))
        for qualifier in ('limited domestic','most output','Only a few','not widespread','dry test conditions','not tested'):
            self.assertIn(qualifier,c['answer'])
        self.assertFalse(c['semantic_support_established'])
        self.assertFalse(c['source_quote_plan']['parsed_text_is_independent_semantic_proof'])
        self.assertEqual(before,catalog)
        plan=validate_qualification_quote_candidate(QUESTION,validated(c),catalog)
        groups=quote_claim_groups(c['answer'],plan,validated(c)['diagnostic_candidate_output']['effective_candidate_output']['citations'])
        self.assertEqual([g['scope_source_index'] for g in groups],[0,1,2])
        self.assertEqual('\n\n'.join(g['claim'] for g in groups),c['answer'])
        for block in plan['blocks']:
            source=catalog[block['image_id']]['source_text_for_exact_quote'];lo,hi=block['source_character_span']
            self.assertEqual(source[lo:hi],block['quote'])

    def test_rewriting_frequency_even_in_both_claim_and_answer_is_rejected(self):
        catalog,request=fixture();c=requested_qualification_quote_candidate(QUESTION,catalog,'region-3',request)
        for replacement in ('Most houses','Many houses','Every house'):
            bad=validated(c);bad[FIELD]['answer']=bad[FIELD]['answer'].replace('Only a few houses',replacement)
            bad[FIELD]['source_quote_plan']['blocks'][0]['quote']=bad[FIELD]['source_quote_plan']['blocks'][0]['quote'].replace('Only a few houses',replacement)
            bad['diagnostic_candidate_output']['effective_candidate_output']['answer']=bad[FIELD]['answer']
            with self.assertRaises(ValueError): validate_qualification_quote_candidate(QUESTION,bad,catalog)

    def test_changed_question_source_identity_or_character_span_is_rejected(self):
        catalog,request=fixture();c=requested_qualification_quote_candidate(QUESTION,catalog,'region-3',request)
        for key in ('source_character_span','answer_character_span','binding_id','original_image_sha256'):
            bad=validated(c);bad[FIELD]['source_quote_plan']['blocks'][0][key]='altered'
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_qualification_quote_candidate(QUESTION,bad,catalog)
        altered=copy.deepcopy(catalog);altered['image-1']['doc_sha256']='b'*64
        with self.assertRaises(ValueError): validate_qualification_quote_candidate(QUESTION,validated(c),altered)
        with self.assertRaises(ValueError): validate_qualification_quote_candidate(QUESTION.replace('fire','water'),validated(c),catalog)

    def test_catalog_order_does_not_change_quotes_but_duplicate_geometry_declines(self):
        catalog,request=fixture();c=requested_qualification_quote_candidate(QUESTION,catalog,'region-3',request)
        reversed_catalog=dict(reversed(list(catalog.items())))
        self.assertEqual(c,requested_qualification_quote_candidate(QUESTION,reversed_catalog,'region-3',request))
        for field,value in (('source_region_type','image'),('image_index',1),('page',1),('parent_evidence_id','region-1')):
            changed=copy.deepcopy(catalog);changed['image-3'][field]=value
            self.assertIsNone(requested_qualification_quote_candidate(QUESTION,changed,'region-3',request))
        changed=copy.deepcopy(catalog);changed['image-1']['source_text_for_exact_quote']='long '*400
        self.assertIsNone(requested_qualification_quote_candidate(QUESTION,changed,'region-3',request))

    def test_disaster_context_cannot_replace_explicit_qualification_source(self):
        catalog,request=fixture();catalog['image-3']['source_text_for_exact_quote']='A fire happened close to an ABC house.'
        self.assertIsNone(requested_qualification_quote_candidate(QUESTION,catalog,'region-3',request))
        catalog,request=fixture();request['query_anchors']=['OTHER']
        self.assertIsNone(requested_qualification_quote_candidate(QUESTION,catalog,'region-3',request))

    def test_effective_answer_citation_and_fabricated_claim_are_rejected(self):
        catalog,request=fixture();c=requested_qualification_quote_candidate(QUESTION,catalog,'region-3',request)
        bad=validated(c);bad['diagnostic_candidate_output']['effective_candidate_output']['answer']+=' extra'
        with self.assertRaises(ValueError): validate_qualification_quote_candidate(QUESTION,bad,catalog)
        bad=validated(c);bad['diagnostic_candidate_output']['effective_candidate_output']['citations'].pop()
        with self.assertRaises(ValueError): validate_qualification_quote_candidate(QUESTION,bad,catalog)
        plan=copy.deepcopy(c['source_quote_plan']);plan['blocks'][0]['quote']='Invented figure title'
        with self.assertRaises(ValueError): quote_claim_groups(c['answer'],plan,validated(c)['diagnostic_candidate_output']['effective_candidate_output']['citations'])
        groups=quote_claim_groups(c['answer'],c['source_quote_plan'],validated(c)['diagnostic_candidate_output']['effective_candidate_output']['citations'])
        import json
        verdict={'claims':[{'id':i,'supported':True,'evidence':[(i+1)%3],'reason':'fixture'} for i in range(3)],
                 'question_answered':True,'reason':'fixture'}
        with self.assertRaises(ValueError): parse_fact_verdict(json.dumps(verdict),groups,3)

    def test_unregistered_context_cannot_create_a_quote_candidate(self):
        with self.assertRaises(ValueError): compose_qualification_quote_candidate(QUESTION,{},object(),{})
        self.assertIsNone(validate_qualification_quote_candidate(QUESTION,{},{}))


if __name__=='__main__': unittest.main()
