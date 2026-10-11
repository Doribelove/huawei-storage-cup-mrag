import json
import tempfile
import unittest
from test_reader_visual_input import prepare_fixture
from evidence_lab.reader import bind_full_source_citations, VisualOutputInvalid

class FullSourceCitationTests(unittest.TestCase):
    def test_full_source_policy_preserves_raw_box_without_claiming_localization_or_support(self):
        with tempfile.TemporaryDirectory() as root:
            prepared,_,_,_=prepare_fixture(root);image=next(iter(prepared.citation_catalog.values()))
            raw={'status':'answered','answer':'synthetic observation','citations':[{'type':'visual_observation',
                'image_id':image['image_id'],'observation':'synthetic observation',
                'region':{'coordinate_system':'selected_image_pixels_top_left','bbox_half_open':[0,0,1,1]}}]}
            result=bind_full_source_citations(json.dumps(raw),prepared)
            box=result['diagnostic_candidate_output']['effective_candidate_output']['citations'][0]['region']['bbox_half_open']
            self.assertEqual(box,[0,0,*image['image']['size_pixels']])
            self.assertEqual(result['citation_region_policy']['remapped_regions'][0]['model_declared_bbox_half_open'],[0,0,1,1])
            self.assertEqual(result['diagnostic_candidate_output']['model_declared_output']['citations'][0]['region']['bbox_half_open'],[0,0,1,1])
            self.assertFalse(result['supported_answer'])
            self.assertFalse(result['citation_region_policy']['precise_fact_localization_verified'])

    def test_coarse_binding_cannot_repair_illegal_boxes_unknown_ids_or_fabricated_quotes(self):
        with tempfile.TemporaryDirectory() as root:
            prepared,_,_,_=prepare_fixture(root);image=next(iter(prepared.citation_catalog.values()))
            citations=[{'type':'visual_observation','image_id':image['image_id'],'observation':'x',
                'region':{'coordinate_system':'selected_image_pixels_top_left','bbox_half_open':[-1,0,1,1]}},
                {'type':'text_quote','image_id':'unknown','quote':'Body A'},
                {'type':'text_quote','image_id':image['image_id'],'quote':'fabricated source text'}]
            for citation in citations:
                with self.assertRaises(VisualOutputInvalid):bind_full_source_citations(json.dumps(
                    {'status':'answered','answer':'x','citations':[citation]}),prepared)

if __name__=='__main__':unittest.main()
