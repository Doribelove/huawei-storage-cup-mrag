import copy
import unittest
from types import SimpleNamespace
from evidence_lab.question_focus import condition_focus, question_focus_sources, recover_question_focus


class QuestionFocusTests(unittest.TestCase):
    def setUp(self):
        self.question='How do copper and iron respond under varying pressure conditions?'
        def row(eid,typ,text,page):
            return {'evidence_id':eid,'document_name':'synthetic_document','doc_sha256':'a'*64,
                    'page':page,'type':typ,'text':text}
        self.old=row('old','text','After surface cleaning, copper and iron were photographed.',1)
        self.body=row('body','text','Pressure was varied for copper and iron; Fig. 7 shows the resulting comparison.',2)
        self.figure=row('figure','image','Figure 7. Copper and iron under pressure conditions.',3)
        self.hits=[self.old,self.body,self.figure];self.rows=copy.deepcopy(self.hits)
    def proposal(self):
        return question_focus_sources(self.question,self.hits,[self.old],self.rows)
    def test_unique_literal_link_with_recorded_original_spans(self):
        proposal,report=self.proposal()
        self.assertEqual(proposal['body']['evidence_id'],'body')
        self.assertEqual(proposal['figure']['evidence_id'],'figure')
        for ref in proposal['literal_body_figure_references']:
            start,end=ref['character_span']
            self.assertEqual(self.body['text'][start:end],ref['text'])
        self.assertFalse(report['proposal_establishes_answer_support'])
    def test_generic_or_multiple_condition_focus_declines(self):
        self.assertIsNone(condition_focus('Which comparison was made under various conditions?'))
        self.assertIsNone(condition_focus('Under hot conditions and under cold conditions, what changed?'))
    def test_whole_word_regular_inflection_without_prefix_guessing(self):
        self.question='How do copper and iron respond under varying load conditions?'
        self.body['text']='Loaded copper and iron are compared in Fig. 7.'
        self.figure['text']='Figure 7. Copper and iron at different loads.'
        self.rows=copy.deepcopy(self.hits)
        self.assertIsNotNone(self.proposal()[0])
        self.figure['text']='Figure 7. Copper and iron in a loadingbay.'
        self.rows=copy.deepcopy(self.hits)
        self.assertIsNone(self.proposal()[0])
    def test_selected_focus_keeps_original_source_priority(self):
        self.old['text']='Copper and iron pressure measurements.'
        self.rows=copy.deepcopy(self.hits)
        self.assertIsNone(self.proposal()[0])
    def test_wrong_figure_reference_and_different_document_decline(self):
        self.body['text']=self.body['text'].replace('Fig. 7','Fig. 8')
        self.rows=copy.deepcopy(self.hits)
        self.assertIsNone(self.proposal()[0])
        self.body['text']=self.body['text'].replace('Fig. 8','Fig. 7')
        self.figure['doc_sha256']='b'*64;self.rows=copy.deepcopy(self.hits)
        self.assertIsNone(self.proposal()[0])
    def test_ambiguous_original_caption_or_body_declines(self):
        duplicate=copy.deepcopy(self.figure);duplicate['evidence_id']='other_figure'
        self.rows.append(duplicate)
        self.assertIsNone(self.proposal()[0])
        self.rows=copy.deepcopy(self.hits)
        duplicate=copy.deepcopy(self.body);duplicate['evidence_id']='other_body'
        self.rows.append(duplicate);self.hits.append(duplicate)
        self.assertIsNone(self.proposal()[0])
    def test_modified_hit_cannot_supply_source_text(self):
        self.body['text']='Pressure for copper and iron; Fig. 7. New unverified facts.'
        with self.assertRaisesRegex(ValueError,'canonical'):
            self.proposal()
    def test_caption_outside_retrieval_pool_cannot_be_selected(self):
        self.hits=self.hits[:-1]
        self.assertIsNone(self.proposal()[0])
    def test_existing_recovery_and_empty_candidate_cannot_add_reader_call(self):
        for generated,prior in (({'answer_candidate_present':True},{'actual_reader_calls':1}),
                                ({'answer_candidate_present':False},{})):
            prepared=SimpleNamespace()
            actual,context,report=recover_question_focus(self.question,generated,prepared,
                hits=self.hits,selected=[self.old],visual=None,reader=None,processor=None,
                budget=None,config={},output=None,prior_recovery=prior)
            self.assertIs(actual,generated);self.assertIs(context,prepared)
            self.assertEqual(report['actual_reader_calls'],0)
            self.assertFalse(report['candidate_applied'])

if __name__=='__main__':unittest.main()
