"""Synthetic provenance/noise regressions; never accuracy measurements."""
import copy
from pathlib import Path
import tempfile
import unittest
from reportlab.pdfgen import canvas
from evidence_lab.lexical import BM25
from evidence_lab.native_pdf import native_tables, SourceVerifier
from evidence_lab.retrieval_policy import CandidatePolicy
from evidence_lab.safe_data import sha256_file
from evidence_lab.semantic import DenseIndex
from test_semantic import row as dense_row, chunk


def row(eid, text='source'):
    return {**dense_row(eid, text), 'type': 'text'}


class NativePolicyTests(unittest.TestCase):
    def fixture(self, folder):
        path = Path(folder)/'native.pdf'
        c = canvas.Canvas(str(path), pagesize=(612,792))
        for x in (50,250,550): c.line(x,600,x,700)
        for y in (600,650,700): c.line(50,y,550,y)
        for x,y,text in [(60,670,'Role'),(260,670,'Responsibility'),
                         (60,620,'Receiving Clerk'),(260,620,'Maintains Diagnostic Report')]:
            c.drawString(x,y,text)
        c.showPage();c.save()
        document = {'sha256':sha256_file(path),'member':'native.pdf','page_count':1}
        return path, document

    def test_native_table_is_reproducible_and_retrievable_with_source_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            path,doc=self.fixture(folder)
            rows=native_tables(path,doc,'native',1,'archive')
            self.assertEqual(rows,native_tables(path,doc,'native',1,'archive'))
            self.assertEqual(len(rows),1)
            self.assertIn('Receiving Clerk\tMaintains Diagnostic Report',rows[0]['text'])
            self.assertEqual(BM25(rows).search('Receiving Clerk')[0]['evidence_id'],rows[0]['evidence_id'])
            class Browser:
                audit={'source_archives':{'data':{'sha256':'archive'}}}
                documents={'native':doc};byid={}
                def checked_page(self,name,page): return self.documents[name]
                def verified_pdf(self,name): return path
            self.assertTrue(SourceVerifier(Browser(),rows).verify(rows[0]['evidence_id'])['all_passed'])
            for field,value in [('text','fabricated'),('bbox_normalized',[0,0,1,1]),('doc_sha256','fake')]:
                tampered=copy.deepcopy(rows[0]);tampered[field]=value
                self.assertFalse(SourceVerifier(Browser(),[tampered]).verify(tampered['evidence_id'])['all_passed'])
            bad=copy.deepcopy(rows[0]);bad['extraction']['cells'][0][0]={'gt_answer':'gold'}
            with self.assertRaises(ValueError): SourceVerifier(Browser(),[bad])

    def test_native_extraction_refuses_changed_pdf_and_bad_page(self):
        with tempfile.TemporaryDirectory() as folder:
            path,doc=self.fixture(folder)
            with self.assertRaises(ValueError): native_tables(path,{**doc,'sha256':'fake'},'native',1,'archive')
            for page in (0,2,True):
                with self.assertRaises(ValueError): native_tables(path,doc,'native',page,'archive')

    def test_noise_policy_retains_numeric_table_values_and_body_text(self):
        records=[{**row('footer','2309'),'bbox_normalized':[.8,.94,.9,.96]},
                 {**row('year','2020'),'bbox_normalized':[.2,.3,.3,.4]},
                 {**row('table','2309'),'type':'table','bbox_normalized':[.2,.94,.3,.96]},
                 row('body','Net income'),row('empty','')]
        policy=CandidatePolicy(records)
        self.assertEqual({r['evidence_id'] for r in policy.searchable()},{'year','table','body'})
        self.assertEqual(policy.excluded['footer'],'page_edge_number')
        self.assertIsNone(policy.report()['answerability_threshold'])

    def test_repeated_edge_text_needs_three_distinct_pages_in_one_document(self):
        records=[{**row(str(i),'Monthly report'),'page':i,'bbox_normalized':[0,.01,1,.05]} for i in (1,2,3)]
        records += [{**row('body','Monthly report'),'bbox_normalized':[0,.4,1,.5]},
                    {**row('other','Monthly report'),'document_name':'other','bbox_normalized':[0,.01,1,.05]}]
        self.assertEqual({r['evidence_id'] for r in CandidatePolicy(records).searchable()},{'body','other'})

    def test_dense_filters_noise_before_top_k_and_all_filtered_is_safe(self):
        with tempfile.TemporaryDirectory() as folder:
            records=[{**row('footer','2309'),'bbox_normalized':[.8,.94,.9,.96]},row('body','Actual source')]
            footer_chunk=chunk('footer1','footer',0);footer_chunk['source_char_span']=[0,4]
            index=DenseIndex(Path(folder)/'dense.json',records,{},[footer_chunk,chunk('body1','body',1)])
            policy=CandidatePolicy(records)
            hits=index.search(chunk('q','body',0)['__vector__'],{},1,candidate_filter=policy.accepts)
            self.assertEqual([h['evidence_id'] for h in hits],['body'])
            self.assertEqual(index.search(chunk('q','body',0)['__vector__'],{},1,candidate_filter=lambda r:False),[])


if __name__=='__main__': unittest.main()
