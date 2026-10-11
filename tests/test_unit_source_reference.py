import copy
import json
import tempfile
import unittest
import pdfplumber
from test_native_unit_prose import printed_fixture,QUESTION
from evidence_lab.native_unit_prose import native_unit_prose_plan
from evidence_lab.unit_source_reference import (requested_unit_reference_candidate,
    native_unit_lines,unit_support_projection,compose_unit_reference_candidate)
from evidence_lab.fact_support import source_support_declaration


def actual_catalog(root):
    path,rows,_=printed_fixture(root)
    result={}
    with pdfplumber.open(path) as pdf:
        for index,page in enumerate(pdf.pages):
            words=page.extract_words();row=rows[index]
            plan=native_unit_prose_plan(QUESTION,words,600,800,
                source_box=row['bbox_points_estimate'],source_transcription='' if index else row['text'])
            image_id='source-'+str(index+1)
            result[image_id]={'image_id':image_id,'document_name':row['document_name'],
                'doc_sha256':row['doc_sha256'],'page':row['page'],'role':'primary_retrieved_region',
                'image':{'size_pixels':[600,800]},'readable_viewport':{'full_page_recovery':True,
                    'native_unit_prose_recovery':True,'native_unit_prose_plan':plan,
                    'native_word_records':words,'viewport_bbox_top_left_points':[0,0,600,800]}}
    return result


def generated(answer,ids=('source-1',)):
    candidate={'answer':answer,'citations':[{'image_id':i,'region':{'bbox_half_open':[0,0,600,800]}} for i in ids]}
    return {'answer_candidate_present':True,'raw_output_for_validation':json.dumps({'answer':answer,'evidence_ids':list(ids)}),
        'diagnostic_candidate_output':{'model_declared_output':{'answer':answer,'evidence_ids':list(ids)},
            'effective_candidate_output':candidate}}


class UnitSourceReferenceTests(unittest.TestCase):
    def test_actual_pdf_word_witness_adds_one_adjacent_supplied_source_without_rewriting(self):
        with tempfile.TemporaryDirectory() as root:
            sources=actual_catalog(root);raw=generated('The layers measure 12 nm, 18 nm and 24 nm.')
            before=copy.deepcopy(raw);original_sources=copy.deepcopy(sources)
            candidate=requested_unit_reference_candidate(QUESTION,raw,sources)
            self.assertEqual(candidate['answer'],raw['diagnostic_candidate_output']['model_declared_output']['answer'])
            self.assertEqual(candidate['evidence_ids'],['source-1','source-2'])
            self.assertFalse(candidate['semantic_support_established'])
            self.assertFalse(candidate['source_acquisition_performed'])
            self.assertEqual(raw,before);self.assertEqual(sources,original_sources)
            for fact in candidate['source_field_plan']['unit_mentions']:
                witness=fact['actual_word_coordinate_witness'];line=witness['literal_source_line']
                words=sources[witness['image_id']]['readable_viewport']['native_word_records']
                self.assertEqual(line['text'],' '.join(words[i]['text'] for i in line['native_word_indices']))

    def test_unseen_units_other_document_ambiguity_and_already_complete_citations_abstain(self):
        with tempfile.TemporaryDirectory() as root:
            sources=actual_catalog(root)
            for value in (generated('The layers measure 12 nm and 77 nm.'),generated('The layers measure 12 nm and 18 nm.'),
                          generated('The layers measure 12 nm, 18 nm and 24 nm.',('source-1','source-2'))):
                self.assertIsNone(requested_unit_reference_candidate(QUESTION,value,sources))
            raw=generated('The layers measure 12 nm, 18 nm and 24 nm.')
            other=copy.deepcopy(sources);other['source-2']['doc_sha256']='different-original-source'
            self.assertIsNone(requested_unit_reference_candidate(QUESTION,raw,other))
            other=copy.deepcopy(sources);other['ambiguous']=copy.deepcopy(other['source-2'])
            self.assertIsNone(requested_unit_reference_candidate(QUESTION,raw,other))

    def test_tampered_quote_and_coordinates_reject_and_partial_citation_does_not_receive_full_page_text(self):
        with tempfile.TemporaryDirectory() as root:
            sources=actual_catalog(root);source=sources['source-1']
            citation={'image_id':'source-1','region':{'bbox_half_open':[0,0,600,800]}}
            projection=unit_support_projection(citation,source)
            declared=source_support_declaration(0,citation,source,QUESTION,'12 nm')
            self.assertEqual(declared['native_unit_prose_lines'],projection['native_unit_prose_lines'])
            self.assertFalse(projection['source_semantic_entity_metric_and_cross_page_association_verified'])
            partial=copy.deepcopy(citation);partial['region']['bbox_half_open']=[300,600,550,750]
            self.assertIsNone(unit_support_projection(partial,source))
            bad=copy.deepcopy(source);bad['readable_viewport']['native_unit_prose_plan']['lines'][0]['text']='Invented source fact'
            with self.assertRaises(ValueError):native_unit_lines(bad)
            bad=copy.deepcopy(source);bad['readable_viewport']['native_unit_prose_plan']['lines'][0]['bbox_top_left_points'][0]+=1
            with self.assertRaises(ValueError):native_unit_lines(bad)

    def test_unregistered_source_context_cannot_compose(self):
        with self.assertRaises(ValueError):compose_unit_reference_candidate(QUESTION,generated('12 nm'),object())


if __name__=='__main__':unittest.main()
