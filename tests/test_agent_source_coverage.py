"""Source coverage/capacity regressions; no relevance or answer labels."""
import unittest
from unittest.mock import Mock
from evidence_lab.agent import (IterativeRetriever,RetrievalBudget,
    retain_source_candidate_pool,select_source_hits,subquery_source_groups)
from evidence_lab.reader import prepare_reader_sources
from evidence_lab.retrieval_policy import CandidatePolicy
from evidence_lab.workbench import Workbench
from test_pilot import fixture


def source(eid,pdf='digital',name=None):
    return fixture(eid,int(eid[1:]) if eid[1:].isdigit() else 0,
        pdf=pdf,name=name or pdf,text='Original source paragraph '+eid+' '*200)


class RankedFixture:
    def __init__(self): self.calls=[]
    def rerank(self,question,hits,top_k,max_pairs):
        self.calls.append({'ids':[h['evidence_id'] for h in hits],'top_k':top_k,'max_pairs':max_pairs})
        ranked=sorted(hits,key=lambda h:(h['doc_sha256']!='digital',h['evidence_id']))
        return ranked[:top_k],{'pairs_encoded':len(hits)*2,'wall_ms':0}


class AgentSourceCoverageTests(unittest.TestCase):
    def diagnostic(self,**budget):
        digital=[source(f'd{i:02}') for i in range(20)];finance=source('f0','finance')
        sources=digital+[finance];calls=[]
        def retrieve(query,document,k):
            calls.append(query)
            rows=([finance]+digital[:19]) if query=='financial results' else digital
            return [{**h,'route':'rrf','score':1} for h in rows[:k]]
        reranker=RankedFixture()
        agent=IterativeRetriever(sources,CandidatePolicy(sources),retrieve,
            lambda eid:{'evidence_id':eid,'all_passed':True},reranker,RetrievalBudget(**budget))
        return agent,reranker,calls

    def test_missing_subquery_document_is_retained_inside_twenty_and_final_five(self):
        agent,reranker,calls=self.diagnostic()
        result=agent.run('financial results and digitalization')
        self.assertEqual(len(calls),3);self.assertEqual(len(reranker.calls),1)
        self.assertEqual(len(reranker.calls[0]['ids']),20)
        self.assertEqual(reranker.calls[0]['top_k'],20)
        self.assertEqual(result['rerank_stats']['pairs_encoded'],40)
        self.assertEqual(result['candidate_pool_selection']['inserted_candidate_ids'],['f0'])
        self.assertEqual(len(result['candidate_pool_selection']['displaced_candidate_ids']),1)
        self.assertEqual(len(result['hits']),5)
        self.assertEqual({h['doc_sha256'] for h in result['hits']},{'digital','finance'})
        self.assertEqual(result['hits'][0]['evidence_id'],'d00')
        self.assertEqual(result['source_selection']['promoted_ids'],['f0'])
        self.assertEqual(result['source_selection']['covered_required_groups'],2)
        self.assertFalse(result['supported_answer']);self.assertFalse(result['answer_generated'])
        self.assertFalse(result['source_selection']['semantic_relevance_established'])
        self.assertLessEqual(len(result['context']['added_regions']),8)
        self.assertTrue(all(c['all_passed'] for c in result['source_binding_checks']))
        self.assertLessEqual(len(prepare_reader_sources(result['hits']+result['context']['added_regions'])),13)

    def test_runtime_k_is_selected_before_return_and_reports_unmet_capacity(self):
        agent,reranker,calls=self.diagnostic()
        one=agent.run('financial results and digitalization',final_regions=1)
        self.assertEqual([h['evidence_id'] for h in one['hits']],['d00'])
        self.assertEqual(one['budget']['final_regions'],1)
        self.assertEqual(one['source_selection']['unmet_source_groups'][0]['reason'],'source_group_not_in_ranked_pool')
        two=agent.run('financial results and digitalization',final_regions=2)
        self.assertEqual({h['doc_sha256'] for h in two['hits']},{'finance','digital'})
        self.assertEqual(two['budget']['final_regions'],2)
        self.assertEqual(agent.budget.final_regions,5)
        for value in [True,0,6,1.5]:
            with self.assertRaises(ValueError):agent.run('Question',final_regions=value)

    def test_small_capacity_keeps_global_best_and_logs_an_unmet_third_group(self):
        ranked=[source('m','main'),source('f','finance'),source('g','energy')]
        routes=[[ranked[0]],[ranked[1]],[ranked[2]]]
        hits,note=select_source_hits(ranked,routes,2)
        self.assertEqual([h['evidence_id'] for h in hits],['m','f'])
        self.assertEqual(note['unmet_source_groups'][0]['reason'],'final_region_capacity')
        self.assertTrue(note['global_best_preserved'])
        self.assertEqual(note['required_groups'],2);self.assertEqual(note['covered_required_groups'],1)

    def test_scoped_and_single_clause_keep_ordinary_order(self):
        rows=[source('a','same'),source('b','same'),source('c','other')]
        for routes,scope in [([[rows[0]],[rows[2]]],'same'),([rows],'')]:
            hits,note=select_source_hits(rows,routes,2,document=scope)
            self.assertEqual(hits,rows[:2]);self.assertEqual(note['promoted_ids'],[])
            pool,note=retain_source_candidate_pool(rows,routes,5,document=scope)
            self.assertEqual(pool,rows)

    def test_identical_pdf_bytes_under_alias_names_do_not_inflate_source_groups(self):
        left=source('a','same','alias_one');right=source('b','same','alias_two')
        self.assertEqual(len(subquery_source_groups([[left],[left],[right]])),1)
        hits,note=select_source_hits([left,right],[[left],[left],[right]],2)
        self.assertEqual(note['unique_pdf_groups'],1)
        self.assertEqual(note['required_groups'],1)

    def test_pure_selection_is_repeatable_and_never_rewrites_source_or_dense_metadata(self):
        import copy
        rows=[source(f'd{i:02}') for i in range(20)]+[source('f0','finance')]
        rows[-1]['matched_chunk']={'__id__':'validated_fixture','source_char_span':[0,5]}
        routes=[[rows[0]],[rows[-1]],[rows[0]]];before=copy.deepcopy((rows,routes))
        first=retain_source_candidate_pool(rows,routes,20)
        self.assertEqual(first,retain_source_candidate_pool(rows,routes,20))
        self.assertEqual(select_source_hits(first[0],routes),select_source_hits(first[0],routes))
        self.assertEqual((rows,routes),before)
        self.assertEqual(first[0][-1]['matched_chunk'],before[0][-1]['matched_chunk'])

    def test_foreign_scoped_source_and_failed_promoted_provenance_are_rejected(self):
        agent,_,_=self.diagnostic()
        with self.assertRaises(ValueError):agent.run('Question','finance')
        agent,_,_=self.diagnostic();agent.verify=lambda eid:{'all_passed':eid!='f0'}
        with self.assertRaises(ValueError):agent.run('financial results and digitalization')
        agent,_,_=self.diagnostic()
        with self.assertRaises(ValueError):agent.run('Question','unknown')

    def test_reranker_cannot_invent_or_duplicate_candidate_ids(self):
        for malicious in [lambda rows:[rows[0],rows[0]],lambda rows:[{**rows[0],'evidence_id':'foreign'}]]:
            agent,_,_=self.diagnostic()
            class Invalid:
                def rerank(self,q,rows,k,p):return malicious(rows),{'pairs_encoded':1,'wall_ms':0}
            agent.reranker=Invalid()
            with self.assertRaises(ValueError):agent.run('financial results and digitalization')

    def test_workbench_passes_requested_k_into_agent_instead_of_slicing_coverage(self):
        row=source('a');workbench=Workbench.__new__(Workbench)
        workbench.calls={'bge_embeddings':0,'reranker_pairs':0}
        workbench.original=Mock(anomalies={});workbench.verify_citation=lambda eid:{'all_passed':True}
        workbench.agent=Mock();workbench.agent.run.return_value={'hits':[row],'context':{},'rerank_stats':None,'budget':{'final_regions':1}}
        value=workbench.search_with_options('Question',top_k=1,mode='agent')
        workbench.agent.run.assert_called_once_with('Question','',final_regions=1)
        self.assertEqual(value['trace']['budget']['final_regions'],1)
        self.assertFalse(value['supported_answer'])


if __name__=='__main__':unittest.main()
