"""Question-only source acquisition boundaries; no benchmark answers."""
import copy
import unittest
from unittest.mock import Mock

from evidence_lab.bounded_selection import (acquire_question_sources, qualification_gap_request,
    required_terms, select_question_pages, supplementary_qualification_source)


def source(eid, page, text, doc='own', kind='text'):
    return {'evidence_id':eid,'doc_sha256':doc,'document_name':doc,'page':page,'text':text,'type':kind}


class BoundedQualificationTests(unittest.TestCase):
    question = 'What drives ABC use for fire-resistant buildings and export needs?'

    def initial(self):
        return [source('export',1,'ABC is used for export.'),
                source('title',2,'ABC export'),
                source('motivation',3,'Recent fires motivate local ABC construction.')]

    def test_compound_search_keeps_hazard_and_property_not_just_captured_suffix(self):
        self.assertEqual(required_terms('Compare XYZ water-proof needs in 2021.'),
                         {'xyz','water','proof','2021'})

    def test_plural_search_cue_prefers_complementary_body_over_redundant_title(self):
        hits = self.initial()
        selected, report = select_question_pages(self.question,hits)
        self.assertEqual([row['evidence_id'] for row in selected], ['export','motivation'])
        self.assertEqual(report['selected_source_cue_forms']['motivation']['fire'], ['fires'])
        self.assertIn('resistant',report['missing_exact_source_search_cues'])
        self.assertTrue(report['cue_forms_are_search_only_not_fact_support'])

    def test_one_focused_query_adds_one_same_document_original_source_without_mutation(self):
        hits = self.initial(); before = copy.deepcopy(hits)
        explicit = source('qualified',4,'ABC is fire resistant under the documented test conditions.')
        backend = Mock(); backend.search.return_value = ([explicit], {'diagnostic_stub':True})
        combined, selected, report = acquire_question_sources(self.question,hits,backend,'rrf_rerank',
                                                              qualification_acquisition=True)
        backend.search.assert_called_once_with('ABC fire','rrf_rerank',5)
        self.assertEqual([row['evidence_id'] for row in selected], ['export','motivation','qualified'])
        self.assertIs(selected[-1],explicit)
        self.assertEqual(hits,before)
        self.assertEqual(combined, hits+[explicit])
        self.assertEqual(report['supplementary_retrieval_count'],1)
        self.assertTrue(report['bounded_qualification_acquisition']['source_binding_and_support_pending'])
        self.assertTrue(report['possible_missing_fields'])

    def test_context_other_documents_images_and_existing_pages_cannot_stand_in_for_new_field(self):
        initial = self.initial(); selected,_ = select_question_pages(self.question,initial)
        request = qualification_gap_request(self.question,selected)
        wrong = [source('disaster',4,'Fires caused damage near ABC construction.'),
                 source('other-doc',4,'ABC is fire resistant.',doc='other'),
                 source('generated-figure-caption',4,'ABC is fire resistant.',kind='image'),
                 source('same-page',1,'ABC is fire resistant.'),
                 source('different-entity',5,'DEF is fire resistant.')]
        self.assertEqual(supplementary_qualification_source(request,wrong,selected),(None,[]))
        backend=Mock();backend.search.return_value=(wrong,{})
        _,actual,report=acquire_question_sources(self.question,initial,backend,'rrf_rerank',
                                                 qualification_acquisition=True)
        self.assertEqual(actual,selected)
        self.assertEqual(backend.search.call_count,1)
        self.assertEqual(report['bounded_qualification_acquisition']['status'],'bounded_search_exhausted')

    def test_existing_literal_field_and_questions_without_anchored_qualification_do_not_trigger(self):
        hits=[source('qualified',1,'ABC is fire resistant and is used for export needs.'),
              source('context',2,'ABC production context.')]
        backend=Mock()
        _,_,report=acquire_question_sources(self.question,hits,backend,'rrf_rerank',qualification_acquisition=True)
        backend.search.assert_not_called()
        self.assertNotIn('bounded_qualification_acquisition',report)
        self.assertIsNone(qualification_gap_request('What changed in the displayed process?',hits))
        self.assertIsNone(qualification_gap_request('What drives fire-resistant building needs?',hits))
        mixed=[hits[0],source('other',3,'Other context.',doc='different')]
        self.assertIsNone(qualification_gap_request(self.question,mixed))

    def test_disabled_feature_retains_one_original_query_pool_expansion(self):
        backend=Mock();backend.search.return_value=(self.initial(),{})
        _,_,report=acquire_question_sources(self.question,self.initial(),backend,'rrf_rerank')
        backend.search.assert_called_once_with(self.question,'rrf_rerank',10)
        self.assertEqual(report['supplementary_retrieval_count'],1)
        self.assertNotIn('bounded_qualification_acquisition',report)


if __name__ == '__main__':
    unittest.main()
