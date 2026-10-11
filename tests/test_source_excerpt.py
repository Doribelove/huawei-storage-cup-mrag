import json
import unittest
from evidence_lab.source_excerpt import long_prose_excerpt

class ExcerptTests(unittest.TestCase):
    def source(self):
        words=[]
        for y,left,text in ((10,0,'Unrelated caption'),(60,0,'The model contains six facets:'),
                            (72,0,'students teachers learning outcomes'),(84,0,'behaviors credibility environment'),
                            (72,400,'Independent right column claim')):
            x=left
            for word in text.split():
                words.append({'text':word,'x0':x,'x1':x+len(word)*4,'top':y,'bottom':y+8});x+=len(word)*4+4
        return {'source_region_type':'text','image':{'size_pixels':[800,1000]},
                'readable_viewport':{'native_word_records':words}}
    def test_literal_text_and_word_coordinates(self):
        source=self.source();r=long_prose_excerpt(source,'model facets','students teachers credibility',[0,0,800,1000])
        self.assertTrue(r['omitted_native_lines']>0)
        for span in r['spans']:
            self.assertEqual(span['text'],' '.join(source['readable_viewport']['native_word_records'][i]['text'] for i in span['native_word_indices']))
        self.assertIn('behaviors credibility environment',[s['text'] for s in r['spans']])
    def test_partial_citation_cannot_expose_outside_text(self):
        self.assertIsNone(long_prose_excerpt(self.source(),'model','students',[0,0,300,200]))
    def test_no_question_or_candidate_anchor(self):
        self.assertIsNone(long_prose_excerpt(self.source(),'volcano','lava',[0,0,800,1000]))
    def test_byte_budget_whole_lines(self):
        r=long_prose_excerpt(self.source(),'model','students',[0,0,800,1000],max_bytes=150)
        self.assertLessEqual(len(json.dumps(r['spans'],ensure_ascii=False,separators=(',',':')).encode()),150)
    def test_non_text_and_malformed_words(self):
        s=self.source();s['source_region_type']='TABLE'
        self.assertIsNone(long_prose_excerpt(s,'model','students',[0,0,800,1000]))
    def test_region_type_case_and_missing_type(self):
        s=self.source();s['source_region_type']='TEXT'
        self.assertIsNotNone(long_prose_excerpt(s,'model','students',[0,0,800,1000]))
        s['source_region_type']=None
        self.assertIsNone(long_prose_excerpt(s,'model','students',[0,0,800,1000]))
        s=self.source();s['readable_viewport']['native_word_records'][0]['x0']=float('nan')
        self.assertIsNone(long_prose_excerpt(s,'model','students',[0,0,800,1000]))
    def test_operator_and_unit_are_unmodified(self):
        s=self.source();s['readable_viewport']['native_word_records'][4]['text']='≥2mg'
        r=long_prose_excerpt(s,'facets','≥2mg',[0,0,800,1000])
        self.assertTrue(any('≥2mg' in line['text'] for line in r['spans']))
