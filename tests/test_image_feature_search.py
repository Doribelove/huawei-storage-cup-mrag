import copy
import unittest

from evidence_lab.image_feature_search import ImageFeatureSearch,source_record_hash


class ImageFeatureSearchTests(unittest.TestCase):
    def fixture(self):
        source={'evidence_id':'original-image','document_name':'OriginalDocument','page':2,
                'doc_sha256':'original-pdf-hash','type':'image','text':'Original caption only'}
        feature={'evidence_id':source['evidence_id'],'document_name':source['document_name'],
            'page':source['page'],'doc_sha256':source['doc_sha256'],'status':'collected',
            'original_source_record_sha256':source_record_hash(source),'page_record_sha256':'page-hash',
            'search_feature_text':'Alpha process labels','native_region_word_count':3,
            'source_quote_or_answer_fact_established':False,'canonical_source_text_replaced':False}
        return source,feature

    def test_matching_added_label_never_replaces_original_caption_with_search_text(self):
        source,feature=self.fixture();before=copy.deepcopy(source)
        result=ImageFeatureSearch([feature],[source]).search('Alpha')[0]
        self.assertEqual(result['canonical_source'],source)
        self.assertNotIn('Alpha',result['canonical_source']['text'])
        self.assertNotIn('search_feature_text',result['canonical_source'])
        self.assertFalse(result['feature_provenance']['source_quote_established'])
        self.assertFalse(result['semantic_support_established'])
        result['canonical_source']['text']='Caller changed a returned copy'
        self.assertEqual(source,before)

    def test_cross_document_tampering_and_claimed_fact_roles_are_rejected(self):
        source,feature=self.fixture()
        changes=[{'document_name':'WrongDocument'},{'page':9},{'doc_sha256':'wrong'},
                 {'original_source_record_sha256':'wrong'},
                 {'source_quote_or_answer_fact_established':True},
                 {'canonical_source_text_replaced':True}]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):
                ImageFeatureSearch([dict(feature,**change)],[source])

    def test_failed_native_acquisition_is_counted_and_cannot_supply_new_search_text(self):
        source,feature=self.fixture();feature['status']='failed'
        index=ImageFeatureSearch([feature],[source])
        self.assertEqual(index.failed_feature_records,1)
        self.assertEqual(index.search('Alpha'),[])

    def native_index(self):
        source,feature=self.fixture()
        feature.update(search_feature_text='Alpha Delta labels',native_region_word_count=1,
            native_words_in_region=[{'text':'Alpha','x0':10,'x1':20,'top':10,'bottom':20}])
        return source,feature,ImageFeatureSearch([feature],[source])

    def test_two_rare_cues_and_native_word_allow_acquisition_without_asserting_support(self):
        source,_,index=self.native_index()
        proposal=index.supplementary_source('What Alpha Delta labels are shown?',[])
        self.assertEqual(proposal['canonical_source'],source)
        self.assertEqual(proposal['acquisition_reason']['native_region_query_cues'],['alpha'])
        self.assertFalse(proposal['acquisition_reason']['support_established'])
        self.assertNotIn('Alpha',proposal['canonical_source']['text'])

    def test_same_pdf_page_alias_and_single_or_generic_cue_do_not_request_new_reader(self):
        source,_,index=self.native_index()
        self.assertIsNone(index.supplementary_source('Alpha Delta',[dict(source,evidence_id='alias')]))
        for question in ('Alpha','What figure is shown over the years?'):
            self.assertIsNone(index.supplementary_source(question,[]))

    def test_caption_only_or_inconsistent_native_records_cannot_supply_new_source(self):
        source,feature,_=self.native_index()
        feature.update(native_words_in_region=[],native_region_word_count=0)
        self.assertIsNone(ImageFeatureSearch([feature],[source]).supplementary_source('Alpha Delta',[]))
        feature['native_region_word_count']=1
        with self.assertRaises(ValueError):
            ImageFeatureSearch([feature],[source]).supplementary_source('Alpha Delta',[])
