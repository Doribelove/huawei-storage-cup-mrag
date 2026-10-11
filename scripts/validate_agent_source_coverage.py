"""Frozen source-only paired Agent probes; never opens supplied QA/answer labels."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch
import psutil
from evidence_lab.agent import (IterativeRetriever,RetrievalBudget,
    retain_source_candidate_pool,select_source_hits,subquery_source_groups)
from evidence_lab.cli import load_evidence
from evidence_lab.demo import EvidenceBrowser
from evidence_lab.lexical import BM25
from evidence_lab.native_pdf import SourceVerifier
from evidence_lab.reader import prepare_reader_sources
from evidence_lab.rerank import LocalReranker
from evidence_lab.retrieval_policy import CandidatePolicy
from evidence_lab.safe_data import sha256_file
from evidence_lab.semantic import LocalBGE,DenseIndex,reciprocal_rank_fusion


BASE_COMMIT='eaf8fb21ebe6abfeb9a0a68268eaa1f16b90cbc2'


def write(path:Path,value):
    if path.exists(): raise ValueError('Completed diagnostic output cannot be overwritten')
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def trusted_baseline(repository:Path,output:Path,expected:str):
    # Only reviewed local code at an exact hash is loaded; document data cannot
    # nominate a module/path/revision. The old source is preserved for review.
    raw=subprocess.check_output(['git','show',BASE_COMMIT+':src/evidence_lab/agent.py'],cwd=repository)
    if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError('Reviewed baseline Agent hash mismatch')
    path=output/'baseline-agent-eaf8fb21.py';path.write_bytes(raw)
    name='evidence_lab._reviewed_baseline_agent'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def coverage(result,routes):
    required={group['doc_sha256'] for group in subquery_source_groups(routes)}
    actual={h['doc_sha256'] for h in result['hits']}
    return {'top_ids':[h['evidence_id'] for h in result['hits']],
        'top_pages':[[h['document_name'],h['page']] for h in result['hits']],
        'top_document_names':sorted({h['document_name'] for h in result['hits']}),
        'unique_original_pdf_groups':len(actual),'required_subquery_groups':len(required),
        'covered_subquery_groups':len(required&actual),
        'subquery_group_coverage':len(required&actual)/len(required) if required else None,
        'all_source_bindings_passed':all(c['all_passed'] for c in result['source_binding_checks']),
        'soft_wall_budget_exceeded':result['soft_wall_budget_exceeded'],
        'queries_executed':result['queries_executed'],'wall_ms':result['wall_ms'],
        'pairs':result['rerank_stats']['pairs_encoded'] if result['rerank_stats'] else 0,
        'dense_verified_matches':sum(bool(n['matched_chunk_verified_against_index_and_source_tokens']) for n in (result['rerank_stats'] or {}).get('source_window_notes',[])),
        'dense_overlap_windows':sum(w['dense_overlap']>0 for n in (result['rerank_stats'] or {}).get('source_window_notes',[]) for w in n['source_windows']),
        'answer_generated':result['answer_generated'],'supported_answer':result['supported_answer']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-repository','audit-dir','phase-dir','full-freeze-dir','pilot-dir','model-dir','reranker-dir','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=Path('configs/agent-coverage-v1.protocol.json'))
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--data-archive',type=Path,help='Explicit relocation only; original audit hash must match')
    args=parser.parse_args()
    if sha256_file(args.protocol)!=args.protocol_sha256: raise ValueError('Frozen protocol byte mismatch')
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    if protocol['base_commit']!=BASE_COMMIT or protocol['version']!='agent-coverage-v1-pre-inference':
        raise ValueError('Unexpected frozen protocol')
    paths={'original_source':args.audit_dir/'regions.jsonl','full_source':args.full_freeze_dir/'regions.jsonl',
        'full_index':args.full_freeze_dir/'full.nano.json','pilot_index':args.pilot_dir/'pilot.nano.json',
        'pilot_manifest':args.baseline_repository/'configs/pilot-v1.manifest.json'}
    for key,path in paths.items():
        if sha256_file(path)!=protocol['input_hashes'][key]: raise ValueError('Frozen source/index hash mismatch: '+key)
    audit=json.loads((args.audit_dir/'data-audit.json').read_text(encoding='utf-8'))
    archive=args.data_archive or Path(audit['source_archives']['data']['path'])
    if not archive.is_file(): raise FileNotFoundError('Original source archive unavailable; supply its current path with --data-archive. No source-verification bypass is permitted: '+str(archive))
    if sha256_file(archive)!=audit['source_archives']['data']['sha256']: raise ValueError('Relocated original archive hash mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    baseline=trusted_baseline(args.baseline_repository,args.output,protocol['baseline_agent_sha256'])
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain']): raise ValueError('Commit clean implementation before inference')
    write(args.output/'run-manifest.json',{'source_commit':source_commit,'baseline_commit':BASE_COMMIT,
        'protocol_sha256':args.protocol_sha256,'input_hashes':protocol['input_hashes'],
        'provided_qa_or_labels_opened':False,'inference_order':'fixed then baseline, not counterbalanced',
        'cache_policy':'Per-case query-route cache shared with baseline; fixed-query misses perform actual BGE inference; no historical logits'})
    original=load_evidence(paths['original_source']);full=load_evidence(paths['full_source'])
    manifest=json.loads(paths['pilot_manifest'].read_text(encoding='utf-8'))
    pilot=[]
    for doc in manifest['documents']:
        pilot.extend(sorted([r for r in original if r['document_name']==doc['name']],key=lambda r:r['evidence_id']))
    pilot_names={d['name'] for d in manifest['documents']}
    pilot_native=[r for r in full if r.get('extraction') and r['document_name'] in pilot_names]
    reports=[];load_times={};embeddings=[];all_certificates={}
    with patch.object(socket.socket,'connect',side_effect=RuntimeError('Source-coverage diagnostic is offline')) as network:
        start=time.perf_counter();model=LocalBGE(args.model_dir);load_times['bge_seconds']=round(time.perf_counter()-start,3)
        browser=EvidenceBrowser(args.audit_dir,args.phase_dir,data_archive=archive,cache_dir=args.output/'citation-cache')
        for corpus,rows,index_rows,index_path in [('pilot',pilot+pilot_native,pilot,paths['pilot_index']),('full',full,full,paths['full_index'])]:
            policy=CandidatePolicy(rows);verifier=SourceVerifier(browser,[r for r in rows if r.get('extraction')]);certificate_cache={}
            start=time.perf_counter();index=DenseIndex(index_path,index_rows,model.identity,source_tokenizer=model.tokenizer)
            load_times[corpus+'_index_seconds']=round(time.perf_counter()-start,3)
            start=time.perf_counter();reranker=LocalReranker(args.reranker_dir,matched_chunk_validator=index.validate_matched_chunk)
            load_times[corpus+'_reranker_seconds']=round(time.perf_counter()-start,3)
            lexical={}
            def verify(eid):
                if eid not in certificate_cache:certificate_cache[eid]=verifier.verify(eid)
                all_certificates[eid]=certificate_cache[eid]
                return certificate_cache[eid]
            print(json.dumps({'stage':'models_loaded','corpus':corpus,'source_regions':len(rows),'dense_chunks':index.chunk_count,'loads':load_times}),flush=True)
            for case in protocol['cases']:
                cache={};route_calls=[];query_statistics=[]
                def retrieve(query,document,k):
                    key=(query,document,k);cached=key in cache
                    route_calls.append({'query':query,'document_scope':document,'k':k,'cached':cached})
                    if not cached:
                        if document not in lexical:lexical[document]=BM25([r for r in policy.searchable() if not document or r['document_name']==document])
                        vector,stats=model.query(query);query_statistics.append(stats)
                        bm25=lexical[document].search(query,k)
                        dense=index.search(vector,model.identity,k,document=document,candidate_filter=policy.accepts)
                        cache[key]=reciprocal_rank_fusion([bm25,dense],k,matched_chunk_validator=index.validate_matched_chunk)
                    return deepcopy(cache[key])
                print(json.dumps({'stage':'case_start','corpus':corpus,'id':case['id']}),flush=True)
                fixed_agent=IterativeRetriever(rows,policy,retrieve,verify,reranker,RetrievalBudget(**protocol['budgets']),matched_chunk_validator=index.validate_matched_chunk)
                fixed=fixed_agent.run(case['question'],case['document_scope'])
                fixed_calls=len(route_calls)
                old_agent=baseline.IterativeRetriever(rows,policy,retrieve,verify,reranker,baseline.RetrievalBudget(**protocol['budgets']),matched_chunk_validator=index.validate_matched_chunk)
                old=old_agent.run(case['question'],case['document_scope'])
                def executed_routes(result):
                    actions=[a for a in result['trace'] if a['operation']=='retrieve_then_verify']
                    return [[{**h,'route':f'planned_query_{number}'} for h in cache[(a['query'],case['document_scope'],protocol['budgets']['route_pool'])]] for number,a in enumerate(actions)]
                fixed_routes=executed_routes(fixed);old_routes=executed_routes(old)
                fused=reciprocal_rank_fusion(fixed_routes,100,matched_chunk_validator=index.validate_matched_chunk)
                repeat_pool=retain_source_candidate_pool(fused,fixed_routes,document=case['document_scope'])
                repeat_final=select_source_hits(fixed['ranked_candidates'],fixed_routes,document=case['document_scope'])
                repeat_ok=(repeat_pool[1]==fixed['candidate_pool_selection'] and repeat_final[0]==fixed['hits']
                    and repeat_final[1]==fixed['source_selection'] and repeat_pool==retain_source_candidate_pool(fused,fixed_routes,document=case['document_scope'])
                    and repeat_final==select_source_hits(fixed['ranked_candidates'],fixed_routes,document=case['document_scope']))
                if not repeat_ok:raise ValueError('Pure source selection repeatability failure')
                fixed_summary=coverage(fixed,fixed_routes);old_summary=coverage(old,old_routes)
                try:
                    clean=prepare_reader_sources(fixed['hits']+fixed['context']['added_regions'])
                    reader_boundary={'status':'bounded_source_contract_passed','regions':len(clean),'characters':sum(len(r['text']) for r in clean),'actual_reader_run':False}
                except ValueError as error:
                    reader_boundary={'status':'explicit_context_rejection','reason':str(error),'actual_reader_run':False}
                result={'corpus':corpus,'case':case,'fixed':fixed_summary,'baseline':old_summary,
                    'fixed_payload':fixed,'baseline_payload':old,'route_calls':route_calls,
                    'fixed_route_call_count':fixed_calls,'actual_bge_embeddings':len(query_statistics),
                    'embedding_statistics':query_statistics,'baseline_queries_all_cached':all(c['cached'] for c in route_calls[fixed_calls:]),
                    'query_routes_equal':fixed_routes==old_routes,'pure_selection_exact_repeat':repeat_ok,
                    'source_group_gain':fixed_summary['covered_subquery_groups']-old_summary['covered_subquery_groups'],
                    'promoted_ids':fixed['source_selection']['promoted_ids'],'displaced_ids':fixed['source_selection']['displaced_ids'],
                    'reader_boundary':reader_boundary,'accuracy_measured':False,
                    'negative_probe_note':'Unsupported-compound query has no independent answerability label; a coverage gain can promote irrelevant sources' if case['id']=='unsupported_compound' else None}
                if not fixed_summary['all_source_bindings_passed'] or not old_summary['all_source_bindings_passed'] or fixed['supported_answer'] or old['supported_answer']:
                    raise ValueError('Source/answer-boundary invariant failure')
                write(args.output/(corpus+'-'+case['id']+'.json'),result);reports.append(result)
                embeddings.extend(query_statistics)
                print(json.dumps({'stage':'case_complete','corpus':corpus,'id':case['id'],
                    'baseline_documents':old_summary['top_document_names'],'fixed_documents':fixed_summary['top_document_names'],
                    'subquery_source_group_gain':result['source_group_gain'],'promoted':result['promoted_ids'],
                    'fixed_pairs':fixed_summary['pairs'],'baseline_pairs':old_summary['pairs'],
                    'fixed_wall_ms':fixed_summary['wall_ms'],'baseline_wall_ms':old_summary['wall_ms'],'reader_boundary':reader_boundary['status']}),flush=True)
            del fixed_agent,old_agent,reranker,index
        unchanged={key:sha256_file(path)==protocol['input_hashes'][key] for key,path in paths.items()}
        memory=psutil.Process().memory_info()
        report={'kind':'Frozen paired source-coverage development probes; not blind/answer accuracy',
            'complete':len(reports)==len(protocol['cases'])*2,'source_commit':source_commit,'protocol_sha256':args.protocol_sha256,
            'case_count':len(reports),'model_identity':model.identity,'load_times':load_times,
            'actual_bge_embeddings':len(embeddings),'total_fixed_reranker_pairs':sum(r['fixed']['pairs'] for r in reports),
            'total_baseline_reranker_pairs':sum(r['baseline']['pairs'] for r in reports),
            'summary':[{k:r[k] for k in ('corpus','case','fixed','baseline','source_group_gain','promoted_ids','displaced_ids','reader_boundary','pure_selection_exact_repeat','query_routes_equal')} for r in reports],
            'all_input_hashes_unchanged':unchanged,'all_source_bindings_passed':all(c['all_passed'] for c in all_certificates.values()),
            'unique_source_bindings':len(all_certificates),'socket_connect_attempts':network.call_count,'paid_api_cost':0,
            'peak_process_working_set_bytes':getattr(memory,'peak_wset',None),'reader_actual_inference':False,'answer_accuracy_measured':False,
            'limits':['Same previously observed development sources, no generalization claim','Distinct source PDF groups do not establish relevance or semantic topic coverage',
                'Source promotion can displace relevant original candidates; all cases/displacements are reported',
                'Different-byte equivalent PDF families are not detected','Baseline shares fixed query cache and warm citation cache; route wall times are not matched end-to-end benchmarks',
                'Fixed first then baseline reranker order is not counterbalanced','No provided30 QA/reference labels or old prediction payload opened','Qwen/vision/AGX unrun']}
        if not report['complete'] or network.call_count or not all(unchanged.values()) or not report['all_source_bindings_passed']:
            raise ValueError('Frozen/offline/source invariants failed')
        write(args.output/'agent-source-coverage-report.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('summary','model_identity','limits')},indent=2),flush=True)


if __name__=='__main__':main()
