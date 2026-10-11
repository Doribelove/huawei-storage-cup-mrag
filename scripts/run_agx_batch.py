"""Resumable local question-only -> retrieval -> native Qwen -> cited support pipeline."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evidence_lab.cli import load_evidence
from evidence_lab.lexical import BM25
from evidence_lab.retrieval_policy import CandidatePolicy
from evidence_lab.reader import LocalQwenReader, prepare_reader_visual_inputs
from evidence_lab.reader_context import ReaderBudget
from evidence_lab.reader_processor import PinnedVisualProcessor, VisualPixelPolicy, prepare_native_visual_inputs
from evidence_lab.native_pdf import SourceVerifier
from evidence_lab.visual_evidence import VisualEvidence
from evidence_lab.safe_data import sha256_file
from evidence_lab.support import QwenSupportVerifier, finalize_answer
from evidence_lab.portable import PortableSource
from evidence_lab.evaluation import load_questions
from evidence_lab.evidence_selection import select_distinct_pages

def save(path,data):
    temp=path.with_suffix('.new')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def memory(torch):
    status=Path('/proc/self/status').read_text().splitlines()
    return {'process_RSS_bytes':next(int(l.split()[1])*1024 for l in status if l.startswith('VmRSS:')),
        'process_lifetime_peak_RSS_bytes':next(int(l.split()[1])*1024 for l in status if l.startswith('VmHWM:')),
        'cuda_allocated_bytes':torch.cuda.memory_allocated(),'cuda_reserved_bytes':torch.cuda.memory_reserved(),
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'note':'AGX unified-memory counters describe overlapping resources; do not add RSS and CUDA allocation'}

def code_version(project):
    files=[]
    for top in ('src','scripts','configs'):
        for path in sorted((project/top).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                files.append({'path':str(path.relative_to(project)),'sha256':sha256_file(path),'bytes':path.stat().st_size})
    return {'received_HEAD':'e34bea8d7b95e0f5049f68897fd38a54068fa404',
            'working_source_manifest_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
            'files':files,'git_repository_present':(project/'.git').is_dir(),'uncommitted_changes':'See source manifest and local change report; package has no .git'}

AUTHORIZATION=json.loads(Path(os.environ['EVIDENCE_LAB_AUTHORIZATION']).read_text())
DEADLINE=AUTHORIZATION['absolute_deadline_epoch']

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--model-dir',type=Path,required=True)
    ap.add_argument('--questions',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--add-empty-evidence-check',action='store_true')
    ap.add_argument('--retry-failed',action='store_true',help='Preserve and retry failed completed attempts with the identical frozen run')
    ap.add_argument('--bge-dir',type=Path);ap.add_argument('--reranker-dir',type=Path)
    ap.add_argument('--native-image-inventory',type=Path,help='Frozen original-PDF search hints; empty-answer recovery only')
    ap.add_argument('--oracle-pages',type=Path,help='Explicit isolated reference-page Reader diagnostic, never a retrieval score')
    args=ap.parse_args();args.data_root=args.data_root.resolve();args.output=args.output.resolve()
    config=json.loads(args.config.read_text())
    allowed_routes=('full_corpus_bm25','bm25','dense','rrf','rrf_rerank','dense_rerank')
    assert config['retrieval_route'] in allowed_routes
    assert config.get('evidence_selection','rank_top2') in ('rank_top2','distinct_page_top2','distinct_page_top3')
    assert 1<=config['selected_regions']<=3 and 1<=config['retrieval_top_k']<=5
    questions=load_questions(args.questions)
    oracle=None
    if args.oracle_pages:
        assert config.get('evaluation_kind')=='oracle_reader_upper_bound'
        oracle_packet=json.loads(args.oracle_pages.read_text())
        assert oracle_packet['schema']=='oracle-reference-pages-only-v1' and oracle_packet['reader_receives_gold_answers'] is False
        oracle={x['sample_id']:x['pages'] for x in oracle_packet['cases']}
        assert set(oracle)=={q['sample_id'] for q in questions}
    if args.add_empty_evidence_check:questions=questions+[{'sample_id':'self_authored_empty_evidence_001','question':'ZXQUNAVAILABLE2099RECOVERYCODE'}]
    args.output.mkdir(parents=True,exist_ok=True)
    previous_summary_path=args.output/'summary.json'
    previous_summary=json.loads(previous_summary_path.read_text()) if previous_summary_path.exists() else None
    version=code_version(Path(__file__).resolve().parents[1])
    feature_search=None;feature_identity=None
    if config.get('bounded_native_image_source_recovery'):
        assert args.native_image_inventory and not oracle,'Native image recovery requires a frozen corpus inventory'
        inventory_path=args.native_image_inventory.resolve()
        inventory=json.loads((inventory_path/'summary.json').read_text())
        features_path=inventory_path/'features.jsonl'
        features_hash=sha256_file(features_path)
        assert features_hash==config['native_image_features_sha256']==inventory['features_sha256']
        assert inventory['corpus_sha256']==sha256_file(args.data_root/'data/full/regions.jsonl')
        assert inventory['completed_attempts']==inventory['image_denominator']
        features=[json.loads(line) for line in features_path.read_text().splitlines()]
        assert len(features)==inventory['image_denominator']
        from evidence_lab.image_feature_search import ImageFeatureSearch
        feature_search=ImageFeatureSearch(features,load_evidence(args.data_root/'data/full/regions.jsonl'))
        feature_identity={'features_sha256':features_hash,'inventory_summary_sha256':sha256_file(inventory_path/'summary.json'),
            'feature_denominator':len(features),'failed_features_retained':feature_search.failed_feature_records,
            'search_hints_are_source_quotes':False,'primary_retrieval_modified':False}
    packages=('torch','torchvision','transformers','tokenizers','numpy','Pillow','nano-vectordb','pypdf','pypdfium2','pdfplumber')
    runtime_versions={name:importlib.metadata.version(name) for name in packages}
    freeze={'schema':'agx-QA-batch-v1','config':config,'config_sha256':sha256_file(args.config),
            'questions_sha256':sha256_file(args.questions),'data_corpus_sha256':sha256_file(args.data_root/'data/full/regions.jsonl'),
            'index_sha256':sha256_file(args.data_root/'data/full/full.nano.json'),'source_version':version,
            'source_members_mapping_sha256':sha256_file(args.data_root/'data/source-members.json'),
            'original_audit_sha256':sha256_file(args.data_root/'data/audit/data-audit.json'),
            'model_revision':'0c351dd01ed87e9c1b53cbc748cba10e6187ff3b','question_ids':[q['sample_id'] for q in questions],
            'gold_answers_or_reference_pages_read':False,'seed':config['seed'],
            'retrieval_route':config['retrieval_route'],'authorization_deadline_epoch':DEADLINE,
            'oracle_reference_pages_used':bool(oracle),'evaluation_kind':config.get('evaluation_kind','retrieved_development_QA'),
            'oracle_reference_projection_sha256':sha256_file(args.oracle_pages) if args.oracle_pages else None,
            'runtime_packages':runtime_versions,'python_version':sys.version,'python_executable':sys.executable,
            'native_image_search_inventory':feature_identity,
            'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH',''),
            'retrieval_model_manifest_hashes':{
                'BGE_conversion':sha256_file(args.bge_dir/'conversion-manifest.json') if args.bge_dir else None,
                'reranker_source':sha256_file(args.reranker_dir/'source-manifest.json') if args.reranker_dir else None}}
    freeze_path=args.output/'freeze.json'
    if freeze_path.exists():assert json.loads(freeze_path.read_text())==freeze,'resume requires identical code/data/model/config'
    else:save(freeze_path,freeze)
    snapshot_root=args.output/'frozen-source'
    if snapshot_root.exists():
        assert code_version(snapshot_root)==version,'Frozen source bytes changed; preserve original code for resume'
    else:
        project_root=Path(__file__).resolve().parents[1]
        for member in version['files']:
            target=snapshot_root/member['path'];target.parent.mkdir(parents=True,exist_ok=True)
            raw=(project_root/member['path']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==member['sha256']
            target.write_bytes(raw)
        assert code_version(snapshot_root)==version
    stored_config=args.output/'freeze-config.json'
    if stored_config.exists():assert sha256_file(stored_config)==freeze['config_sha256']
    else:stored_config.write_bytes(args.config.read_bytes())
    if previous_summary is not None:
        history=args.output/'invocation-history';history.mkdir(exist_ok=True)
        save(history/f'previous-summary-{time.time_ns()}.json',previous_summary)
    build=json.loads((args.data_root/'data/full/full-build-report.json').read_text())
    assert freeze['data_corpus_sha256']==build['frozen_corpus_sha256'] and freeze['index_sha256']==build['index_sha256']
    import torch
    import numpy as np
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);torch.set_num_threads(4)
    gpu_free,gpu_total=torch.cuda.mem_get_info()
    save(args.output/'startup-resources.json',{'epoch':time.time(),'GPU_name':torch.cuda.get_device_name(0),
        'CUDA_free_bytes':gpu_free,'CUDA_total_bytes':gpu_total,'resources':memory(torch)})
    assert gpu_free>=22*1024**3,'Insufficient CUDA headroom for the fixed full model'
    rows=load_evidence(args.data_root/'data/full/regions.jsonl');policy=CandidatePolicy(rows);bm25=BM25(policy.searchable())
    browser=PortableSource(args.data_root);verifier=SourceVerifier(browser,[r for r in rows if r.get('extraction')]);visual=VisualEvidence(verifier)
    batch_started=time.perf_counter();reader=None;native_processor=None;load_seconds=None;cases=[]
    budget=ReaderBudget()
    with patch.object(socket.socket,'connect',side_effect=RuntimeError('Inference batch forbids network')) as network:
        def write_summary():
            saved_load_path=args.output/'model-load.json'
            saved_load=json.loads(saved_load_path.read_text()) if saved_load_path.exists() else {}
            save(args.output/'summary.json',{'schema':'agx-QA-summary-v1','question_denominator':len(questions),
                'completed_attempts':len(cases),'failed':sum(c['status']=='failed' for c in cases),
                'source_recovery_errors':sum(bool(c.get('bounded_source_recovery',{}).get('error')) for c in cases),
                'source_support_errors':sum(c.get('evidence_support',{}).get('status')=='verifier_failed'
                    or bool(c.get('evidence_support',{}).get('error')) for c in cases),
                'source_support_attempt_errors':sum(bool(p.get('error')) for c in cases
                    for p in c.get('evidence_support',{}).get('passes',[])),
                'output_contract_failures':sum(bool(c.get('output_validation_error')) for c in cases),
                'released_by_automated_support_gate':sum(c['supported_answer'] for c in cases),
                'refused_or_withheld':sum(c['status']=='not_enough_evidence' for c in cases),
                'official_answer_accuracy':None,'official_judge_executed':False,'held_out_test':False,
                'network_attempts':network.call_count,'model_load_seconds':saved_load.get('seconds',load_seconds),
                'elapsed_batch_seconds':time.perf_counter()-batch_started,
                'elapsed_scope':'This invocation only; original attempts and previous invocation summaries retained',
                'sum_recorded_question_seconds':sum(c['end_to_end_seconds'] for c in cases),
                'resume_invocation':any('resumed_from' in c for c in cases),
                'cases':[{'sample_id':c['sample_id'],'status':c['status'],'supported_answer':c['supported_answer'],
                    'result_file':str(Path(c['attempt_directory'])/'result.json')} for c in cases]})
        backend=None
        if config['retrieval_route'] not in ('full_corpus_bm25','bm25'):
            assert args.bge_dir and args.reranker_dir,'Pinned retrieval model paths are required'
            from evidence_lab.retrieval_backend import SourceRetrievalBackend
            backend=SourceRetrievalBackend(args.data_root,args.bge_dir,args.reranker_dir)
            save(args.output/'retrieval-models.json',backend.identity)
        for question in questions:
            sample_id=question['sample_id'];assert sample_id and all(c.isalnum() or c in '_-' for c in sample_id)
            case_dir=args.output/sample_id
            completed=sorted(case_dir.glob('attempt-*/result.json')) if case_dir.exists() else []
            if completed and json.loads(completed[-1].read_text()).get('completed'):
                cached=json.loads(completed[-1].read_text())
                retryable=(cached['status']=='failed' or bool(cached.get('output_validation_error'))
                    or cached.get('evidence_support',{}).get('status')=='verifier_failed')
                if not retryable or not args.retry_failed:
                    cached['resumed_from']=str(completed[-1]);cases.append(cached);continue
            count=len(list(case_dir.glob('attempt-*'))) if case_dir.exists() else 0
            attempt=case_dir/f'attempt-{count+1:03d}';attempt.mkdir(parents=True,exist_ok=False)
            result={'sample_id':sample_id,'question':question['question'],'attempt':count+1,'completed':False,
                    'status':'running','supported_answer':False,'answer':'','citations':[],
                    'official_answer_correctness':'not_judged','started_epoch':time.time(),'attempt_directory':str(attempt)}
            save(attempt/'result.json',result);start=time.perf_counter();stage='retrieval';stage_started=start;load_for_this_attempt=0
            torch.cuda.reset_peak_memory_stats()
            try:
                assert time.time()<DEADLINE-240,'not enough authorized time for another question'
                t=time.perf_counter()
                if oracle is not None:
                    pages=oracle[sample_id];assert 1<=len(pages)<=config['selected_regions'],'Oracle context exceeds Reader capacity; do not omit evidence'
                    hits=[]
                    for page in pages:
                        assert set(page)=={'document_name','page'}
                        scope=[r for r in policy.searchable() if r['document_name']==page['document_name'] and r['page']==page['page']]
                        assert scope,'Oracle source page has no canonical searchable evidence'
                        ranked=BM25(scope).search(question['question'],1)
                        assert ranked,'No source selected on requested oracle page'
                        hits.append(ranked[0])
                    retrieval_details={'oracle_reference_page_diagnostic':True,'gold_answer_read_by_reader':False,
                        'full_retrieval_success_claimed':False,'reference_pages':pages}
                elif backend is None:hits=bm25.search(question['question'],config['retrieval_top_k']);retrieval_details={}
                else:hits,retrieval_details=backend.search(question['question'],config['retrieval_route'],config['retrieval_top_k'])
                selected=(select_distinct_pages(hits,config['selected_regions'])
                    if config.get('evidence_selection') in ('distinct_page_top2','distinct_page_top3') else hits[:config['selected_regions']])
                selection_report=None
                if config.get('question_source_selection'):
                    from evidence_lab.bounded_selection import select_question_pages
                    selected,selection_report=select_question_pages(question['question'],hits,config['selected_regions'])
                    if backend is not None and selection_report['possible_missing_fields']:
                        initial_hits=hits
                        hits,additional=backend.search(question['question'],config['retrieval_route'],10)
                        selected,after=select_question_pages(question['question'],hits,config['selected_regions'])
                        selection_report={'initial':selection_report,'after_single_pool_expansion':after,
                            'supplementary_retrieval_count':1,'initial_evidence_ids':[h['evidence_id'] for h in initial_hits],
                            'expanded_top_k':10,'additional_details':additional,
                            'possible_missing_fields':after['possible_missing_fields']}
                retrieval={'route':'oracle_source_page_diagnostic' if oracle else config['retrieval_route'],'hits':hits,'seconds':time.perf_counter()-t,
                           'selection':config.get('evidence_selection','rank_top2'),
                           'reference_labels_used_for_selection':False,'details':retrieval_details,
                           'selected_evidence_ids':[h['evidence_id'] for h in selected]}
                if selection_report is not None:retrieval['question_only_selection']=selection_report
                save(attempt/'retrieval.json',retrieval);result['retrieval_seconds']=retrieval['seconds']
                if not hits:
                    result.update(completed=True,status='not_enough_evidence',refusal_reason='retrieval_returned_no_evidence',
                        actual_reader_run=False,evidence_support={'status':'no_evidence','semantic_verifier_run':False,'accepted':False})
                else:
                    stage='source_binding';stage_started=time.perf_counter()
                    source_proofs=[verifier.verify(hit['evidence_id']) for hit in selected]
                    assert all(p['all_passed'] for p in source_proofs)
                    save(attempt/'source-proofs.json',source_proofs)
                    artifacts={}
                    for i,hit in enumerate(selected):
                        folder=attempt/f'evidence-{i+1}';visual.prepare_region(hit['evidence_id'],folder,dpi=config['render_dpi'],
                            readable_question=question['question'] if config.get('readable_pdf_windows') else None)
                        artifacts[hit['evidence_id']]=folder
                    result['source_binding_seconds']=time.perf_counter()-stage_started
                    stage='model_load';stage_started=time.perf_counter()
                    if reader is None:
                        load_start=time.perf_counter()
                        native_processor=PinnedVisualProcessor(args.model_dir,policy=VisualPixelPolicy(
                            min_pixels=config['min_pixels'],max_pixels=config['max_pixels']),runtime_profile='agx-r35-5-qualified-r2')
                        reader=LocalQwenReader(args.model_dir,device='cuda',dtype='float16')
                        load_seconds=time.perf_counter()-load_start
                        load_for_this_attempt=load_seconds
                        assert all(p.device.type=='cuda' and p.dtype==torch.float16 for p in reader.model.parameters())
                        assert reader.model.config._attn_implementation=='eager'
                        save(args.output/'model-load.json',{'seconds':load_seconds,'identity':reader.identity,
                            'parameter_count':sum(p.numel() for p in reader.model.parameters()),'resources':memory(torch),
                            'quantized':False,'CPU_offload':False,'torch_compile':False})
                    torch.cuda.reset_peak_memory_stats()
                    stage='native_input_preparation';stage_started=time.perf_counter()
                    prepared=prepare_reader_visual_inputs(question['question'],hits,visual=visual,
                        primary_artifacts=artifacts,tokenizer=reader.context_tokenizer,budget=budget,max_new_tokens=config['max_new_tokens'],
                        prompt_projection=config.get('prompt_projection','original_whole_regions_v1'))
                    save(attempt/'prepared-context.json',{'messages':prepared.messages,'image_catalog':prepared.citation_catalog,
                        'source_preparation':prepared.report,'native_input_or_generation_completed':False})
                    native=prepare_native_visual_inputs(prepared,native_processor,budget=budget,max_new_tokens=config['max_new_tokens'])
                    tensors={name:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.cpu().contiguous().numpy().tobytes()).hexdigest()} for name,v in native.tensors.items()}
                    save(attempt/'input.json',{'messages':prepared.messages,'image_catalog':prepared.citation_catalog,
                        'source_preparation':prepared.report,'native_preprocessing':native.report,'tensor_metadata':tensors,
                        'input_token_ids':native.tensors['input_ids'][0].tolist(),'image_grid_thw':native.tensors['image_grid_thw'].tolist()})
                    result['native_input_preparation_seconds']=time.perf_counter()-stage_started
                    stage='generation_and_output_binding';stage_started=time.perf_counter()
                    if native.report['budget_validation'].get('status') == 'blocked_expanded_prompt_budget':
                        stage='native_expanded_prompt_budget'
                        raise RuntimeError(json.dumps(native.report['budget_validation'], ensure_ascii=False))
                    result['reader_generation_attempted']=True
                    generated=reader.read_visual(native,budget=budget,
                        max_new_tokens=config['max_new_tokens'],max_seconds=config['generation_seconds'],
                        full_source_citations=config.get('visual_citation_policy')=='full_selected_image_after_raw_binding_v1',
                        compact_source_citations=config.get('compact_source_citations',False),
                        answer_source_ids=config.get('answer_source_ids',False))
                    save(attempt/'reader-output.json',generated)
                    result['actual_reader_run']=generated['actual_reader_run']
                    result['reader_generation_and_binding_seconds']=time.perf_counter()-stage_started
                    try:
                        raw_candidate=json.loads(generated.get('raw_output_for_validation',''))
                        raw_answer=raw_candidate.get('answer','') if isinstance(raw_candidate,dict) and (
                            config.get('answer_source_ids') or raw_candidate.get('status')=='answered') else ''
                        result['raw_candidate_answer_for_isolated_evaluation']=raw_answer if isinstance(raw_answer,str) else ''
                    except json.JSONDecodeError:result['raw_candidate_answer_for_isolated_evaluation']=''
                    if config.get('bounded_original_page_recovery'):
                        from evidence_lab.source_recovery import recover_empty_reader
                        stage='bounded_source_recovery';stage_started=time.perf_counter()
                        generated,prepared,recovery= recover_empty_reader(
                            question['question'],generated,prepared,hits=hits,selected=selected,
                            visual=visual,reader=reader,processor=native_processor,budget=budget,
                            config=config,output=attempt/'source-recovery',feature_search=feature_search)
                        result['bounded_source_recovery']=recovery
                    if config.get('source_parameter_field_candidates'):
                        from evidence_lab.source_calculation import compose_source_parameter_candidate
                        stage='source_parameter_fields'
                        original_generated=generated
                        generated=compose_source_parameter_candidate(question['question'],generated,prepared)
                        if generated is not original_generated:
                            save(attempt/'source-field-candidate.json',generated)
                            result['source_literal_candidate_record']=str(attempt/'source-field-candidate.json')
                    if config.get('source_period_field_candidates'):
                        from evidence_lab.source_calculation import compose_source_period_candidate
                        stage='source_period_fields'
                        original_generated=generated
                        generated=compose_source_period_candidate(question['question'],generated,prepared)
                        if generated is not original_generated:
                            save(attempt/'source-period-candidate.json',generated)
                            result['source_period_candidate_record']=str(attempt/'source-period-candidate.json')
                    stage='semantic_support';stage_started=time.perf_counter()
                    support_class=QwenSupportVerifier
                    if config.get('fact_support'):
                        from evidence_lab.fact_support import QwenFactSupportVerifier
                        support_class=QwenFactSupportVerifier
                    support=support_class(reader,native_processor).verify(
                        question['question'],generated,prepared,attempt/'support',max_new_tokens=config['support_max_new_tokens'],
                        max_seconds=config['support_seconds'])
                    final=finalize_answer(generated,support)
                    if config.get('generic_metric_guard') and generated.get('answer_candidate_present'):
                        from evidence_lab.semantic_sanity import metric_sanity
                        diagnostic=generated['diagnostic_candidate_output']
                        check=metric_sanity(question['question'],diagnostic.get('effective_candidate_output',diagnostic['model_declared_output']))
                        final['generic_metric_sanity']=check
                        if not check['passed']:final.update(status='not_enough_evidence',answer='',citations=[],supported_answer=False,
                            presentation_status='withheld_invalid_metric_substitution')
                    result['semantic_support_seconds']=time.perf_counter()-stage_started
                    result.update(final,completed=True,actual_reader_run=True)
            except Exception as exc:
                result.update(status='failed',completed=True,failure_stage=stage,error={'type':type(exc).__name__,
                    'message':str(exc),'traceback':traceback.format_exc()},supported_answer=False,answer='',citations=[],
                    failure_stage_elapsed_seconds=time.perf_counter()-stage_started)
            try:result['resources']=memory(torch)
            except Exception as exc:result['resource_measurement_error']={'type':type(exc).__name__,'message':str(exc)}
            result['end_to_end_seconds']=time.perf_counter()-start
            result['cold_model_load_seconds_in_this_attempt']=load_for_this_attempt
            result['finished_epoch']=time.time();save(attempt/'result.json',result)
            save(attempt/'timing.json',{'seconds_through_fsynced_result_write':time.perf_counter()-start,
                'diagnostic_seconds_before_result_write':result['end_to_end_seconds'],
                'cold_model_load_seconds':load_for_this_attempt,'official_scoring_time_verified':False})
            cases.append(result)
            print(json.dumps({'sample_id':sample_id,'status':result['status'],'supported_answer':result['supported_answer'],
                'candidate_present':result.get('answer_candidate_present',False),'seconds':result['end_to_end_seconds'],
                'failure_stage':result.get('failure_stage'),'error':result.get('error',{}).get('message')},ensure_ascii=False),flush=True)
            write_summary()
        assert network.call_count==0
        # Recover a missing/stale summary even when every result is cached.
        write_summary()

if __name__=='__main__':main()
