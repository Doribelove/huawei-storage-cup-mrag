"""Small arithmetic/storage fixtures; these are not model retrieval measurements."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from evidence_lab.safe_data import FORBIDDEN_EVIDENCE_FIELDS
from evidence_lab.semantic import DenseIndex,reciprocal_rank_fusion,token_windows,valid_vector


def row(eid,text='source'):
    return {'role':'source_pdf_region','evidence_id':eid,'text':text,'doc_sha256':'pdf-unit-fixture','document_name':'unit','page':1}


def chunk(cid,parent,axis):
    vector=np.zeros(1024,dtype=np.float32);vector[axis]=1
    return {'__id__':cid,'__vector__':vector,'parent_evidence_id':parent,
        'source_token_span':[0,1],'source_char_span':[0,6],'tokens_with_specials':3}


class SemanticTests(unittest.TestCase):
    def test_windows_cover_tail_and_never_exceed_budget(self):
        for count in [0,1,510,511,958,20000]:
            windows=token_windows(count,510,64)
            self.assertTrue(all(0<=s<e<=count and e-s<=510 for s,e in windows))
            self.assertEqual(set(range(count)),{i for s,e in windows for i in range(s,e)})
        for args in [(1,0,0),(1,510,510),(-1,510,64)]:
            with self.assertRaises(ValueError): token_windows(*args)

    def test_query_vectors_reject_nonfinite_zero_and_wrong_dimension(self):
        for vector in [np.zeros(1024),np.ones(3),np.full(1024,np.nan),np.full(1024,np.inf)]:
            with self.assertRaises(ValueError): valid_vector(vector)
        self.assertAlmostEqual(float(np.linalg.norm(valid_vector(np.ones(1024)))),1)

    def test_dense_persistence_dedup_and_same_dimension_model_mismatch(self):
        identity={'unit_fixture_model':'arithmetic-only','revision':'a','dimension':1024}
        records=[row('a'),row('b')]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'fixture.json'
            index=DenseIndex(path,records,identity,[chunk('a1','a',0),chunk('a2','a',0),chunk('b1','b',1)])
            query=chunk('q','a',0)['__vector__']
            hits=index.search(query,identity,2)
            self.assertEqual([h['evidence_id'] for h in hits],['a','b'])
            self.assertEqual(hits[0]['matched_chunk']['__id__'],'a1')
            reloaded=DenseIndex(path,records,identity)
            self.assertEqual(reloaded.search(query,identity,2),hits)
            with self.assertRaises(ValueError): index.search(query,{**identity,'revision':'b'},2)
            with self.assertRaises(ValueError): DenseIndex(path,records,{**identity,'revision':'b'})
            with self.assertRaises(ValueError): DenseIndex(path,[row('a','changed'),row('b')],identity)
            with self.assertRaises(ValueError): DenseIndex(path,records,identity,[chunk('new','a',0)])

    def test_dense_rejects_gold_and_unknown_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'fixture.json'
            for record in [{**row('a'),'gt_answer':'gold'},{**row('a'),'extra':{'answer':'gold'}},{**row('a'),'role':'historical_image_analysis'}]:
                with self.assertRaises(ValueError): DenseIndex(path,[record],{},[chunk('a1','a',0)])
            self.assertFalse(path.exists())

    def test_dense_scope_is_applied_before_ranking_and_empty_scope_is_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'fixture.json'
            rows=[row('a'),{**row('b'),'document_name':'other'},{**row('empty',''),'document_name':'empty'}]
            index=DenseIndex(path,rows,{},[chunk('a1','a',0),chunk('b1','b',1)])
            vector=chunk('q','a',0)['__vector__']
            self.assertEqual([h['evidence_id'] for h in index.search(vector,{},1,document='other')],['b'])
            self.assertEqual(index.search(vector,{},1,document='empty'),[])
            with self.assertRaises(ValueError): index.search(vector,{},1,document='unknown')

    def test_chunks_reject_top_level_and_nested_labels_before_any_output(self):
        for field in sorted(FORBIDDEN_EVIDENCE_FIELDS):
            for payload in [{field:'SYNTHETIC_GOLD'},{'metadata':{'details':[{field:'SYNTHETIC_GOLD'}]}}]:
                with self.subTest(payload=payload),tempfile.TemporaryDirectory() as temp:
                    path=Path(temp)/'fixture.json'
                    # Put the forbidden chunk last so a valid prefix cannot create partial output.
                    chunks=[chunk('good','a',0),{**chunk('bad','a',1),**payload}]
                    with self.assertRaisesRegex(ValueError,'QA/answer fields cannot enter evidence adapter'):
                        DenseIndex(path,[row('a')],{},chunks)
                    self.assertEqual(list(Path(temp).iterdir()),[])

    def test_rejected_chunks_do_not_rewrite_existing_index(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'fixture.json'
            DenseIndex(path,[row('a')],{},[chunk('good','a',0)])
            before=path.read_bytes()
            for payload in [{'gt_answer':'SYNTHETIC_GOLD'},{'metadata':[{'answer':'SYNTHETIC_GOLD'}]}]:
                with self.subTest(payload=payload):
                    with self.assertRaises(ValueError):
                        DenseIndex(path,[row('a')],{},[{**chunk('bad','a',1),**payload}])
                    self.assertEqual(path.read_bytes(),before)
                    self.assertEqual(list(Path(temp).iterdir()),[path])

    def test_write_and_reload_reject_identical_label_payloads_without_rewrite(self):
        import json
        for payload in [{'gt_answer':'SYNTHETIC_GOLD'},{'metadata':{'details':[{' GT_ANSWER ':'SYNTHETIC_GOLD'}]}}]:
            with self.subTest(payload=payload),tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'fixture.json'
                with self.assertRaisesRegex(ValueError,'QA/answer fields cannot enter evidence adapter'):
                    DenseIndex(path,[row('a')],{},[{**chunk('bad','a',0),**payload}])
                self.assertFalse(path.exists())
                DenseIndex(path,[row('a')],{},[chunk('good','a',0)])
                # Alter only this arithmetic test fixture to exercise the existing read-side guard.
                storage=json.loads(path.read_text(encoding='utf-8'));storage['data'][0].update(payload)
                path.write_text(json.dumps(storage),encoding='utf-8');before=path.read_bytes()
                with self.assertRaisesRegex(ValueError,'QA/answer fields cannot enter evidence adapter'):
                    DenseIndex(path,[row('a')],{})
                self.assertEqual(path.read_bytes(),before)
                self.assertEqual(list(Path(temp).iterdir()),[path])

    def test_rrf_source_whitelist_rank_math_and_deterministic_tie(self):
        a={**row('a'),'route':'bm25','score':900,'irrelevant_debug':'discard'}
        b={**row('b'),'route':'bm25','score':8}
        dense_a={**row('a'),'route':'bge_dense','score':0.1}
        dense_b={**row('b'),'route':'bge_dense','score':0.9}
        result=reciprocal_rank_fusion([[a,b],[dense_b,dense_a]],2)
        self.assertEqual([h['evidence_id'] for h in result],['a','b'])
        self.assertAlmostEqual(result[0]['score'],1/61+1/62,places=8)
        self.assertNotIn('irrelevant_debug',result[0])
        self.assertEqual(len(reciprocal_rank_fusion([[a,a]],2)[0]['route_ranks']),1)
        self.assertEqual(reciprocal_rank_fusion([[],[]]),[])

    def test_rrf_rejects_conflicting_sources_and_gold(self):
        hit={**row('a'),'route':'bm25','score':1}
        with self.assertRaises(ValueError): reciprocal_rank_fusion([[hit],[{**hit,'text':'different'}]])
        with self.assertRaises(ValueError): reciprocal_rank_fusion([[{**hit,'gt_answer':'gold'}]])
        with self.assertRaises(ValueError): reciprocal_rank_fusion([[hit]*101])


if __name__=='__main__': unittest.main()
