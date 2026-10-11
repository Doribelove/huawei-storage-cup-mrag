"""Bounded iterative source retrieval. Documents cannot supply executable tool calls."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass,replace
import re
import time
from .pilot import ContextAssembler
from .safe_data import evidence_output
from .semantic import reciprocal_rank_fusion


@dataclass(frozen=True)
class RetrievalBudget:
    max_queries: int = 3
    max_seconds: float = 45
    route_pool: int = 20
    final_regions: int = 5
    max_rerank_pairs: int = 40
    max_added_regions: int = 8
    max_added_characters: int = 6000

    def __post_init__(self):
        bounds={'max_queries':(1,3),'route_pool':(5,20),'final_regions':(1,5),
                'max_rerank_pairs':(1,40),'max_added_regions':(0,8),'max_added_characters':(0,6000)}
        for name,(low,high) in bounds.items():
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or not low<=value<=high:
                raise ValueError('Agent retrieval budget exceeded')
        if isinstance(self.max_seconds,bool) or not isinstance(self.max_seconds,(int,float)) or not 1<=self.max_seconds<=120:
            raise ValueError('Agent wall budget exceeded')


def plan_queries(question: str, max_queries: int = 3) -> list[str]:
    if not isinstance(question,str) or not question.strip() or len(question)>500 or not 1<=max_queries<=3:
        raise ValueError('Invalid bounded question plan')
    # A transparent deterministic decomposition, not an LLM planner or relevance oracle.
    queries=[question.strip()];seen={question.strip().casefold()}
    clauses=re.split(r'\s+(?:and|versus|vs\.?)\s+|[;；]',question,flags=re.I)
    if len(clauses)>1:
        for clause in clauses:
            clause=clause.strip(' ?？.,，:：')
            if len(clause)>=6 and re.search(r'[A-Za-z\u3400-\u9fff]',clause) and clause.casefold() not in seen:
                queries.append(clause);seen.add(clause.casefold())
            if len(queries)==max_queries: break
    return queries[:max_queries]


class EvidenceGraph:
    """Exact document/page membership and original-order edges; no inferred fact edges."""
    def __init__(self,records:list[dict],policy):
        self.records={r['evidence_id']:evidence_output(r) for r in records}
        if len(self.records)!=len(records): raise ValueError('Duplicate graph evidence ID')
        self.policy=policy;self.pages=defaultdict(list);self.original_groups=defaultdict(list);self.edges=defaultdict(list)
        for row in self.records.values():
            if row.get('role')!='source_pdf_region': raise ValueError('Graph source role mismatch')
            key=(row['doc_sha256'],row['document_name'],row['page']);self.pages[key].append(row['evidence_id'])
            aliases=[a for a in row.get('aliases',[]) if a.get('document_name')==row['document_name']]
            if len(aliases)==1 and 'content_list_member' in aliases[0] and 'record_index' in aliases[0]:
                alias=aliases[0]
                self.original_groups[(*key,alias['content_list_member'])].append((alias['record_index'],row['evidence_id']))
        for values in self.original_groups.values():
            values.sort()
            for left,right in zip(values,values[1:]):
                self.edges[left[1]].append({'target':right[1],'relation':'next_original_region_on_same_pdf_page'})
                self.edges[right[1]].append({'target':left[1],'relation':'previous_original_region_on_same_pdf_page'})
        self.assembler=ContextAssembler(records)

    def validate_hit(self,hit:dict):
        row=evidence_output({k:v for k,v in hit.items() if k not in ('score','route','route_ranks')})
        if row['evidence_id'] not in self.records or row!=self.records[row['evidence_id']]:
            raise ValueError('Retrieved evidence/graph source binding mismatch')
        if not self.policy.accepts(row): raise ValueError('Source noise cannot enter the agent candidates')

    def page(self,document_name:str,page:int)->list[dict]:
        if isinstance(page,bool) or not isinstance(page,int) or page<1:
            raise ValueError('Invalid source page lookup')
        matches=[key for key in self.pages if key[1]==document_name and key[2]==page]
        if len(matches)!=1: raise ValueError('Unknown/ambiguous original source page')
        return [self.records[eid] for eid in self.pages[matches[0]]]

    def stats(self)->dict:
        return {'source_nodes':len(self.records),'original_page_nodes':len(self.pages),
                'original_order_edges':sum(map(len,self.edges.values())),
                'edge_meaning':'Original same-page source order, not semantic entailment',
                'arbitrary_file_or_shell_tool':False}


def subquery_source_groups(routes:list[list[dict]],document:str='')->list[dict]:
    """Executed subquery top1 PDF identities, not relevance/answer labels."""
    if document: return []
    groups=[];seen=set()
    for number,route in enumerate(routes[1:],1):
        if not route: continue
        row=route[0];identity=row['doc_sha256']
        if identity in seen: continue
        seen.add(identity)
        groups.append({'query_index':number,'anchor_evidence_id':row['evidence_id'],
                       'doc_sha256':identity,'document_name':row['document_name']})
    return groups


def retain_source_candidate_pool(fused:list[dict],routes:list[list[dict]],limit:int=20,*,
                                 document:str='',final_limit:int=5)->tuple[list[dict],dict]:
    """Reserve existing subquery source groups inside the same reranking pool budget."""
    if isinstance(limit,bool) or not isinstance(limit,int) or not 5<=limit<=20:
        raise ValueError('Agent candidate pool budget exceeded')
    if isinstance(final_limit,bool) or not isinstance(final_limit,int) or not 1<=final_limit<=5:
        raise ValueError('Agent final source budget exceeded')
    required=subquery_source_groups(routes,document);original=fused[:limit]
    reserved=[];ids=set()
    if fused:
        reserved.append(fused[0]);ids.add(fused[0]['evidence_id'])
    # k=1 preserves the ordinary global result; two source groups cannot fit it.
    if final_limit>1:
        for group in required:
            representative=next((h for h in fused if h['doc_sha256']==group['doc_sha256']),None)
            if representative is not None and representative['evidence_id'] not in ids:
                reserved.append(representative);ids.add(representative['evidence_id'])
    for row in fused:
        if len(reserved)>=limit: break
        if row['evidence_id'] not in ids:
            reserved.append(row);ids.add(row['evidence_id'])
    # Keep original fused order and metadata, including validated matched_chunk.
    selected=[h for h in fused if h['evidence_id'] in ids]
    before={h['evidence_id'] for h in original}
    return selected,{'required_subquery_source_groups':required,
        'original_candidate_ids':[h['evidence_id'] for h in original],
        'selected_candidate_ids':[h['evidence_id'] for h in selected],
        'inserted_candidate_ids':[h['evidence_id'] for h in selected if h['evidence_id'] not in before],
        'displaced_candidate_ids':[h['evidence_id'] for h in original if h['evidence_id'] not in ids],
        'candidate_limit':limit,'source_group_identity':'original PDF SHA256',
        'semantic_relevance_established':False}


def select_source_hits(ranked:list[dict],routes:list[list[dict]],limit:int=5,*,
                       document:str='')->tuple[list[dict],dict]:
    """Keep global best, then source representatives, with explicit capacity failures."""
    if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=5:
        raise ValueError('Agent final source budget exceeded')
    required=subquery_source_groups(routes,document);selected=[];ids=set();groups=set();unmet=[]
    if ranked:
        selected.append(ranked[0]);ids.add(ranked[0]['evidence_id']);groups.add(ranked[0]['doc_sha256'])
    for group in required:
        identity=group['doc_sha256']
        if identity in groups: continue
        representative=next((h for h in ranked if h['doc_sha256']==identity),None)
        if representative is None:
            unmet.append({**group,'reason':'source_group_not_in_ranked_pool'});continue
        if len(selected)>=limit:
            unmet.append({**group,'reason':'final_region_capacity'});continue
        selected.append(representative);ids.add(representative['evidence_id']);groups.add(identity)
    for row in ranked:
        if len(selected)>=limit: break
        if row['evidence_id'] not in ids:
            selected.append(row);ids.add(row['evidence_id']);groups.add(row['doc_sha256'])
    selected=[h for h in ranked if h['evidence_id'] in ids]
    original=ranked[:limit];original_ids={h['evidence_id'] for h in original}
    required_ids={g['doc_sha256'] for g in required}
    return selected,{'policy':'bounded_executed_subquery_source_groups',
        'required_subquery_source_groups':required,'unmet_source_groups':unmet,
        'ordinary_ranked_top_ids':[h['evidence_id'] for h in original],
        'selected_ids':[h['evidence_id'] for h in selected],
        'promoted_ids':[h['evidence_id'] for h in selected if h['evidence_id'] not in original_ids],
        'displaced_ids':[h['evidence_id'] for h in original if h['evidence_id'] not in ids],
        'required_groups':len(required_ids),'covered_required_groups':len(required_ids&groups),
        'unique_pdf_groups':len(groups),'final_region_limit':limit,
        'global_best_preserved':not ranked or selected[0]['evidence_id']==ranked[0]['evidence_id'],
        'source_group_identity':'original PDF SHA256','semantic_relevance_established':False}


class IterativeRetriever:
    def __init__(self,records:list[dict],policy,retrieve,verify,reranker=None,budget:RetrievalBudget|None=None,*,matched_chunk_validator=None):
        self.graph=EvidenceGraph(records,policy);self.retrieve=retrieve;self.verify=verify
        self.reranker=reranker;self.budget=budget or RetrievalBudget()
        self.matched_chunk_validator=matched_chunk_validator

    def run(self,question:str,document:str='',*,final_regions:int|None=None)->dict:
        if not isinstance(document,str) or (document and not any(r['document_name']==document for r in self.graph.records.values())):
            raise ValueError('Unknown Agent source document scope')
        if final_regions is not None and (isinstance(final_regions,bool) or not isinstance(final_regions,int) or not 1<=final_regions<=self.budget.final_regions):
            raise ValueError('Requested final source budget exceeded')
        budget=self.budget if final_regions is None else replace(self.budget,final_regions=final_regions)
        started=time.perf_counter();deadline=started+budget.max_seconds
        queries=plan_queries(question,budget.max_queries);trace=[];routes=[];seen=set();certificates={};stop=[]
        for number,query in enumerate(queries):
            if time.perf_counter()>=deadline: stop.append('wall_budget_before_next_query');break
            action_started=time.perf_counter()
            hits=self.retrieve(query,document,budget.route_pool)
            if len(hits)>budget.route_pool: raise ValueError('Retriever exceeded candidate budget')
            for hit in hits:
                self.graph.validate_hit(hit)
                if document and hit['document_name']!=document: raise ValueError('Retriever violated Agent document scope')
            new={hit['evidence_id'] for hit in hits}-seen
            for hit in hits:
                eid=hit['evidence_id']
                if eid not in certificates:
                    certificates[eid]=self.verify(eid)
                    if not certificates[eid].get('all_passed'): raise ValueError('Agent candidate source provenance failed')
            routes.append([{**hit,'route':f'planned_query_{number}'} for hit in hits]);seen.update(new)
            trace.append({'operation':'retrieve_then_verify','query':query,'new_source_regions':len(new),
                          'candidate_regions':len(hits),'source_documents':sorted({h['document_name'] for h in hits}),
                          'wall_ms':round((time.perf_counter()-action_started)*1000,3),
                          'reflection':'Source provenance checked; semantic answer support remains unestablished'})
            if number>0 and not new: stop.append('no_new_source_evidence');break
        if len(routes)==len(queries): stop.append('plan_exhausted')
        # Fusion already sees at most3x20 source-verified candidates. Retain its full
        # arithmetic ranking before enforcing the unchanged20-region inference pool.
        fused=reciprocal_rank_fusion(routes,100,matched_chunk_validator=self.matched_chunk_validator) if routes else []
        candidates,pool_selection=retain_source_candidate_pool(fused,routes,budget.route_pool,
            document=document,final_limit=budget.final_regions)
        candidate_ids={h['evidence_id'] for h in candidates}
        rerank_stats=None
        if self.reranker is not None and candidates and time.perf_counter()<deadline:
            candidates,rerank_stats=self.reranker.rerank(question,candidates,budget.route_pool,
                                                       budget.max_rerank_pairs)
            trace.append({'operation':'local_cross_encoder_rerank','pairs':rerank_stats['pairs_encoded'],
                          'wall_ms':rerank_stats['wall_ms'],'answer_support_established':False})
        elif self.reranker is not None and candidates:
            stop.append('wall_budget_before_reranking')
        ranked_ids=set()
        for row in candidates:
            self.graph.validate_hit(row)
            if row['evidence_id'] not in candidate_ids or row['evidence_id'] in ranked_ids:
                raise ValueError('Reranker changed the Agent candidate identity contract')
            ranked_ids.add(row['evidence_id'])
            if 'matched_chunk' in row:
                if self.matched_chunk_validator is None: raise ValueError('Unverified Agent Dense match')
                self.matched_chunk_validator(row)
        hits,source_selection=select_source_hits(candidates,routes,budget.final_regions,document=document)
        trace.append({'operation':'bounded_final_source_selection',**source_selection})
        context=self.graph.assembler.assemble(hits,max_added=budget.max_added_regions,
            max_characters=budget.max_added_characters,candidate_filter=self.graph.policy.accepts)
        # Preserve a top source candidate from each executed subquery within the same
        # context budget. Reranking the complete question can otherwise erase one topic.
        context_ids={h['evidence_id'] for h in hits}|{h['evidence_id'] for h in context['added_regions']}
        for number,route in enumerate(routes[1:],1):
            if not route: continue
            source=self.graph.records[route[0]['evidence_id']]
            if source['evidence_id'] in context_ids: continue
            if len(context['added_regions'])>=budget.max_added_regions or context['added_characters']+len(source['text'])>budget.max_added_characters:
                context['stop_reasons'].append('subquery_context_budget');continue
            context['added_regions'].append(source);context['added_characters']+=len(source['text'])
            context_ids.add(source['evidence_id'])
            context['links'].append({'added_evidence_id':source['evidence_id'],
                                    'relation':'top_source_candidate_for_executed_subquery','query_index':number})
        # Read-only original-page lookup is scoped to retrieved evidence, never arbitrary paths.
        pages=[]
        for document_name,page in sorted({(h['document_name'],h['page']) for h in hits}):
            regions=self.graph.page(document_name,page)
            pages.append({'document_name':document_name,'page':page,'source_regions_on_page':len(regions),
                          'unrepresented_source_assets':[{'evidence_id':r['evidence_id'],'type':r['type']} for r in regions
                              if r['type'] in ('table','image','equation') and not r['text'].strip()]})
        for row in context['added_regions']:
            eid=row['evidence_id']
            if eid not in certificates: certificates[eid]=self.verify(eid)
            if not certificates[eid].get('all_passed'): raise ValueError('Agent graph context provenance failed')
        elapsed=time.perf_counter()-started
        return {'question':question,'document_scope':document,'plan':queries,'queries_executed':len(routes),
                'hits':hits,'context':context,'source_page_lookups':pages,'trace':trace,
                'stop_reasons':sorted(set(stop+['reader_unavailable'])),
                'source_binding_checks':list(certificates.values()),'graph':self.graph.stats(),
                'rerank_stats':rerank_stats,'candidate_pool_selection':pool_selection,'source_selection':source_selection,
                'ranked_candidates':candidates,'wall_ms':round(elapsed*1000,3),'budget':budget.__dict__,
                'wall_limit_enforced_between_operations':True,'soft_wall_budget_exceeded':elapsed>budget.max_seconds,
                'supported_answer':False,'answer_generated':False,'support_state':'requires_reader',
                'plan_kind':'deterministic bounded clause decomposition; no LLM planning has run',
                'limits':['Source-valid candidates are not a correct answer','No calibrated relevance/refusal threshold',
                          'Reserved source groups come only from executed subquery top1; they can be irrelevant',
                          'Distinct-byte equivalent PDF families are not identified by exact SHA grouping',
                          'Per-operation bounded inference can finish after the soft wall deadline']}
