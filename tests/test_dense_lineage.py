"""Long original-span/security regressions with arithmetic vectors, not model accuracy."""
import copy
import re
import tempfile
from pathlib import Path
import unittest
import numpy as np
from evidence_lab.semantic import DenseIndex,reciprocal_rank_fusion,token_windows
from evidence_lab.rerank import LocalReranker,select_passage_windows
from test_semantic import row


class WordTokenizer:
    """Explicit synthetic whitespace offsets; no claim of BGE token equivalence."""
    def __call__(self,text,**kwargs):
        spans=[m.span() for m in re.finditer(r'\S+',text)]
        return {'input_ids':list(range(len(spans))),'offset_mapping':spans}
    def num_special_tokens_to_add(self,pair=False): return 4 if pair else 2


def source_fixture(folder):
    text=' '.join(['early']*100+['unrelated']*80);source=row('long',text)
    identity={'unit_fixture_model':'arithmetic-only','max_length':40,'overlap_tokens':8,'dimension':1024}
    tokenizer=WordTokenizer();offsets=tokenizer(text)['offset_mapping'];chunks=[]
    spans=token_windows(len(offsets),38,8)
    for number,(start,end) in enumerate(spans):
        vector=np.zeros(1024,dtype=np.float32);vector[0 if number==len(spans)-1 else 1]=1
        chunks.append({'__id__':f'long:tokens:{start}:{end}','parent_evidence_id':'long','__vector__':vector,
            'source_token_span':[start,end],
            'source_char_span':[0 if start==0 else offsets[start][0],len(text) if end==len(offsets) else offsets[end-1][1]],
            'tokens_with_specials':end-start+2})
    index=DenseIndex(Path(folder)/'dense.json',[source],identity,chunks,source_tokenizer=tokenizer)
    query=np.zeros(1024,dtype=np.float32);query[0]=1
    return source,index,index.search(query,identity,1)[0],offsets


class DenseLineageTests(unittest.TestCase):
    def test_rrf_keeps_tail_match_and_two_original_windows_despite_early_lexical_bias(self):
        with tempfile.TemporaryDirectory() as folder:
            source,index,dense,offsets=source_fixture(folder)
            lexical={**source,'route':'bm25','score':100}
            fused=reciprocal_rank_fusion([[lexical],[dense]],1,matched_chunk_validator=index.validate_matched_chunk)[0]
            self.assertEqual(fused['matched_chunk'],dense['matched_chunk'])
            match=index.validate_matched_chunk(fused)['source_char_span']
            selected,omitted=select_passage_windows(source['text'],offsets,40,'early',match,2)
            control,_=select_passage_windows(source['text'],offsets,40,'early',None,2)
            self.assertGreater(omitted,0);self.assertEqual(len(selected),2)
            self.assertTrue(all(w['dense_overlap']>0 for w in selected))
            self.assertEqual(max(w['source_char_span'][1] for w in selected),len(source['text']))
            self.assertTrue(all(w['source_char_span'][1]<match[0] for w in control))
            for w in selected:
                s,e=w['source_token_span'];a,b=w['source_char_span']
                self.assertEqual(source['text'][a:b],' '.join(source['text'].split()[s:e]))

    def test_fusion_requires_validator_and_rejects_in_bounds_forged_or_cross_parent_match(self):
        with tempfile.TemporaryDirectory() as folder:
            source,index,dense,_=source_fixture(folder)
            with self.assertRaisesRegex(ValueError,'Unverified'): reciprocal_rank_fusion([[dense]])
            variants=[]
            for field,value in [('__id__','unknown'),('source_char_span',[0,10]),('source_token_span',[0,1]),('tokens_with_specials',3),('verified',True)]:
                forged=copy.deepcopy(dense);forged['matched_chunk'][field]=value;variants.append(forged)
            variants += [{**dense,'evidence_id':'other'},{**dense,'text':source['text']+' altered'}]
            for forged in variants:
                with self.subTest(forged=forged),self.assertRaises(ValueError):
                    reciprocal_rank_fusion([[forged]],matched_chunk_validator=index.validate_matched_chunk)

    def test_invalid_duplicate_cannot_hide_behind_route_deduplication(self):
        with tempfile.TemporaryDirectory() as folder:
            _,index,dense,_=source_fixture(folder);forged=copy.deepcopy(dense)
            forged['matched_chunk']['source_char_span']=[0,1]
            with self.assertRaises(ValueError):
                reciprocal_rank_fusion([[dense,forged]],matched_chunk_validator=index.validate_matched_chunk)

    def test_output_mutation_does_not_rewrite_canonical_registry_or_other_routes(self):
        with tempfile.TemporaryDirectory() as folder:
            _,index,dense,_=source_fixture(folder)
            fused=reciprocal_rank_fusion([[dense]],matched_chunk_validator=index.validate_matched_chunk)[0]
            original=copy.deepcopy(dense['matched_chunk']);fused['matched_chunk']['source_char_span'][0]=0
            self.assertEqual(dense['matched_chunk'],original)
            self.assertEqual(index.validate_matched_chunk(dense),original)
            with self.assertRaises(ValueError): index.validate_matched_chunk(fused)

    def test_multiple_valid_matches_use_best_input_rank_then_canonical_id(self):
        with tempfile.TemporaryDirectory() as folder:
            source,index,dense,_=source_fixture(folder)
            cid=min(index.chunk_metadata);parent,tokens,chars,size=index.chunk_metadata[cid]
            early={**dense,'matched_chunk':{'__id__':cid,'source_token_span':list(tokens),'source_char_span':list(chars),'tokens_with_specials':size}}
            a=reciprocal_rank_fusion([[dense],[early]],matched_chunk_validator=index.validate_matched_chunk)[0]
            b=reciprocal_rank_fusion([[early],[dense]],matched_chunk_validator=index.validate_matched_chunk)[0]
            self.assertEqual(a['matched_chunk'],b['matched_chunk'])
            self.assertEqual(a['matched_chunk']['__id__'],min(cid,dense['matched_chunk']['__id__']))
            distractor={**row('distractor'),'route':'bm25','score':1}
            best=reciprocal_rank_fusion([[dense],[distractor,early]],1,matched_chunk_validator=index.validate_matched_chunk)[0]
            self.assertEqual(best['matched_chunk'],dense['matched_chunk'])

    def test_stored_span_must_match_actual_original_tokenizer_geometry(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            _,index,dense,_=source_fixture(folder);path=Path(folder)/'dense.json'
            storage=json.loads(path.read_text(encoding='utf-8'))
            record=next(c for c in storage['data'] if c['__id__']==dense['matched_chunk']['__id__'])
            record['source_char_span'][0]+=1;path.write_text(json.dumps(storage),encoding='utf-8')
            corrupt=DenseIndex(path,list(index.records.values()),index.identity,source_tokenizer=WordTokenizer())
            query=np.zeros(1024,dtype=np.float32);query[0]=1
            with self.assertRaisesRegex(ValueError,'original tokenizer window'): corrupt.search(query,index.identity)

    def test_malformed_metadata_is_rejected_before_persistence(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'new.json';vector=np.ones(1024,dtype=np.float32)
            bad={'__id__':'bad','parent_evidence_id':'a','__vector__':vector,'source_token_span':[0,1],'source_char_span':[0,99],'tokens_with_specials':3}
            with self.assertRaises(ValueError): DenseIndex(path,[row('a')],{},[bad])
            self.assertFalse(path.exists())

    def test_reranker_never_uses_unverified_external_span_before_inference(self):
        class NoInference:
            def get_num_threads(self): return 4
            def tensor(self,*args,**kwargs): raise AssertionError('Unverified match reached inference')
        with tempfile.TemporaryDirectory() as folder:
            _,index,dense,_=source_fixture(folder)
            reranker=LocalReranker.__new__(LocalReranker);reranker.torch=NoInference()
            reranker.tokenizer=WordTokenizer();reranker.identity={'threads':4};reranker.max_length=64
            reranker.matched_chunk_validator=None
            with self.assertRaisesRegex(ValueError,'Unverified'): reranker.rerank('query',[dense])
            reranker.matched_chunk_validator=index.validate_matched_chunk;forged=copy.deepcopy(dense)
            forged['matched_chunk']['source_char_span']=[0,1]
            with self.assertRaisesRegex(ValueError,'bound dense index'): reranker.rerank('query',[forged])


if __name__=='__main__': unittest.main()
