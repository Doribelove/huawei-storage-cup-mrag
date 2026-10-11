import copy
import math
import unittest

from evidence_lab.native_image_features import image_region_features


class NativeImageFeatureTests(unittest.TestCase):
    def source(self):
        return {'type':'image','evidence_id':'original-id','document_name':'original-doc',
            'doc_sha256':'original-pdf-sha','page':1,'text':'Parsed caption',
            'bbox_points_estimate':[10,10,100,100],
            'bbox_coordinate_system':'original quantized region'}

    def word(self,text,x0,top,x1,bottom):
        return dict(text=text,x0=x0,top=top,x1=x1,bottom=bottom)

    def extract(self,words,source=None):
        return image_region_features(source or self.source(),words,width=200,height=300,
                                     page_record_sha256='verified-page-record')

    def test_adjacent_figure_words_and_boundary_crossing_glyphs_are_not_region_text(self):
        inside=self.word('Alpha',20,20,40,30)
        other=self.word('OtherDiagram',110,20,150,30)
        crossing=self.word('Border',95,40,110,50)
        result=self.extract([inside,other,crossing])
        self.assertEqual(result['native_words_in_region'],[inside])
        self.assertNotIn('OtherDiagram',result['search_feature_text'])
        self.assertNotIn('Border',result['search_feature_text'])

    def test_caption_cue_does_not_fill_missing_raster_labels_or_become_answer_quote(self):
        caption=self.word('Figure 7. Process',20,105,90,115)
        result=self.extract([caption])
        self.assertEqual(result['native_region_word_count'],0)
        self.assertTrue(result['no_native_region_text_visual_acquisition_needed'])
        self.assertEqual(result['nearby_caption_candidates_for_search_only'],['Figure 7. Process'])
        self.assertFalse(result['source_quote_or_answer_fact_established'])
        self.assertFalse(result['caption_is_answer_sentence'])

    def test_source_identity_and_original_source_text_remain_unchanged(self):
        source=self.source();before=copy.deepcopy(source)
        result=self.extract([self.word('Literal',20,20,40,30)],source)
        self.assertEqual(source,before)
        self.assertEqual(result['original_parsed_image_text'],source['text'])
        self.assertFalse(result['canonical_source_text_replaced'])
        self.assertEqual(result['evidence_id'],source['evidence_id'])
        self.assertFalse(result['semantic_cell_or_graph_node_association_verified'])

    def test_invalid_region_or_native_geometry_is_rejected(self):
        for box in ([0,0,210,100],[10,10,10,100],[math.nan,0,100,100]):
            source=dict(self.source(),bbox_points_estimate=box)
            with self.subTest(box=box),self.assertRaises(ValueError):
                self.extract([],source)
        with self.assertRaises(ValueError):
            self.extract([self.word('Impossible',-5,0,5,10)])
