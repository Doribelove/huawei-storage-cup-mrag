import unittest
from evidence_lab.integrity import link_image_chunk,page_fallback


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.path='mineru-parsed/example/auto/images/figure.jpg'
        self.doc={'example':{'sha256':'source-hash','member':'original_pdf_files_100/example_origin.pdf'}}
        self.region={'role':'source_pdf_region','document_name':'example','doc_sha256':'source-hash',
            'evidence_id':'e1','page':3,'source_bbox':[1,2,30,40],'image_member':self.path,
            'text':'Figure 1. Test','aliases':[{'pdf_member':self.doc['example']['member']}]}
        self.chunk={'__id__':'c1','full_doc_id':'missing-id','file_path':'example.pdf',
                    'content':f'Image Path: {self.path}\nCaptions: Figure 1. Test\nVisual Analysis: unsupported statement'}

    def test_asset_link_does_not_verify_generated_analysis_or_rewrite_id(self):
        row=link_image_chunk(self.chunk,[self.region],self.doc,lambda _: 'image-hash')
        self.assertEqual(row['link_status'],'asset_linked')
        self.assertEqual(row['content_verification'],'unknown')
        self.assertFalse(row['historical_analysis_usable_as_verified_evidence'])
        self.assertEqual(row['original_full_doc_id'],'missing-id')
        self.assertEqual(row['source_links'][0]['page'],3)
        self.assertTrue(row['source_links'][0]['caption_exact_after_whitespace_normalization'])
        self.assertNotIn('content',row)

    def test_other_document_or_wrong_pdf_hash_cannot_supply_a_link(self):
        for change in ({'document_name':'other'},{'doc_sha256':'other-hash'},
                       {'aliases':[{'pdf_member':'other.pdf'}]}):
            with self.subTest(change=change):
                row=link_image_chunk(self.chunk,[{**self.region,**change}],self.doc,lambda _: 'image-hash')
                self.assertEqual(row['link_status'],'unknown')
                self.assertEqual(row['source_links'],[])

    def test_missing_ambiguous_or_unsafe_image_is_unknown(self):
        cases=[({'content':'no image reference'},[self.region]),
               ({},[self.region,self.region]),
               ({'content':'Image Path: ../escape.jpg'},[self.region])]
        for change,regions in cases:
            with self.subTest(change=change,count=len(regions)):
                self.assertEqual(link_image_chunk({**self.chunk,**change},regions,self.doc,lambda _: 'hash')['link_status'],'unknown')
        def missing(_): raise KeyError('absent asset')
        self.assertEqual(link_image_chunk(self.chunk,[self.region],self.doc,missing)['link_status'],'unknown')

    def test_fallback_retains_invalid_bbox_and_has_no_guessed_text(self):
        review={'doc_sha256':'hash','document_name':'example','page':2,'pdf_member':'example.pdf',
                'visual_classification':'uncertain','review_note':'invalid original coordinates',
                'render_path':'page.png','render_sha256':'render-hash',
                'anomalies':[{'kind':'invalid_bbox','source_bbox':[0,51,682,1001]}]}
        row=page_fallback(review)
        self.assertEqual(row['source_anomalies'][0]['source_bbox'],[0,51,682,1001])
        self.assertEqual(row['bbox_normalized'],[0,0,1,1])
        self.assertEqual(row['role'],'source_pdf_page')
        self.assertFalse(row['automatically_searchable'])
        self.assertEqual(row['text'],'')
        with self.assertRaises(ValueError): page_fallback({**review,'visual_classification':'guessed'})


if __name__=='__main__': unittest.main()
