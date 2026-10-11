"""One original-page reread after an empty answer from cropped evidence.

The first model output and all first-pass artifacts remain unchanged. Recovery
is source preparation and another candidate, never a support authorization.
"""
import hashlib
import json
import time
import traceback
from pathlib import Path


def _empty_recovery_kind(generated, prepared, allow_empty_known_references=False, *, require_crop=True):
    if generated.get('answer_candidate_present') is not False:
        return None
    def unique_keys(pairs):
        value={}
        for key,item in pairs:
            if key in value:
                raise ValueError('Duplicate JSON key')
            value[key]=item
        return value
    text=generated.get('raw_output_for_validation','')
    if not isinstance(text,str) or len(text)>16000:
        return None
    try:
        raw=json.loads(text,object_pairs_hook=unique_keys)
    except (ValueError,TypeError):
        return None
    if (not isinstance(raw,dict) or set(raw)!={'answer','evidence_ids'}
            or raw['answer']!='' or not isinstance(raw['evidence_ids'],list)):
        return None
    identifiers=raw['evidence_ids']
    kind=None
    if (not identifiers
            and generated.get('structure_validation',{}).get('status')=='passed'
            and generated.get('source_binding_validation',{}).get('status')=='passed'):
        kind='valid_empty_answer'
    elif (allow_empty_known_references is True and 1<=len(identifiers)<=8
            and all(isinstance(i,str) and i in prepared.citation_catalog for i in identifiers)
            and generated.get('actual_reader_run') is True
            and generated.get('vision_inference_performed') is True
            and generated.get('generation_provenance')=='fixed_local_weights_native_visual_batch'
            and generated.get('output_validation_error',{}).get('stage')=='structure'):
        # This original output remains invalid. It authorizes no candidate and
        # only requests a new original-page input and a strictly validated call.
        # The exact parsed schema establishes the empty-answer/reference fault;
        # an exception's formatted message is retained as a diagnostic, not a
        # condition for acquisition of another source view.
        kind='invalid_empty_answer_with_known_source_references'
    if kind is None:
        return None
    cropped=any(source['role']=='primary_retrieved_region'
               and source.get('image',{}).get('file')=='crop.png'
               and source.get('readable_viewport',{}).get('status')=='selected_native_pdf_window'
               for source in prepared.citation_catalog.values())
    return kind if cropped or not require_crop else None


def recovery_eligible(generated, prepared, *, allow_empty_known_references=False):
    return _empty_recovery_kind(generated,prepared,allow_empty_known_references) is not None


def _save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def recover_empty_reader(question, generated, prepared, *, hits, selected,
                         visual, reader, processor, budget, config, output, feature_search=None):
    kind=_empty_recovery_kind(generated,prepared,config.get('bounded_empty_known_reference_recovery',False),
                              require_crop=False)
    proposal=None
    if kind and config.get('bounded_native_image_source_recovery') and feature_search is not None:
        proposal=feature_search.supplementary_source(question,selected)
    if proposal is None:
        kind=_empty_recovery_kind(generated,prepared,config.get('bounded_empty_known_reference_recovery',False))
    report={'schema':'bounded-original-source-recovery-v3',
            'eligible':kind is not None,'recovery_request_kind':kind,'attempt_limit':1,
            'reader_call_limit':1,'actual_reader_calls':0,'recovery_candidate_used':False,
            'retrieval_source_pages_changed':False,'reference_pages_used':False,
            'semantic_support_established':False,
            'original_validation_error_preserved':generated.get('output_validation_error'),
            'original_failure_reclassified_as_valid':False,
            'original_reader_output_sha256':hashlib.sha256(
                generated.get('raw_output_for_validation','').encode()).hexdigest()}
    if not report['eligible']:
        return generated,prepared,report
    from .reader import prepare_reader_visual_inputs
    from .reader_processor import prepare_native_visual_inputs
    directory=Path(output);directory.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    original=generated
    try:
        artifacts={};changes=[]
        old_by_parent={s['parent_evidence_id']:s for s in prepared.citation_catalog.values()
                       if s['role']=='primary_retrieved_region'}
        recovery_selected=[proposal['canonical_source']] if proposal else list(selected)
        recovery_hits=list(recovery_selected) if proposal else list(hits)
        continuation_added=False
        if proposal:
            report.update(retrieval_source_pages_changed=True,acquisition_proposal=proposal,
                recovery_route='rare_native_image_cues',
                reason=f'{kind}; one new original image region with rare literal query cues; new candidate requires independent binding and support')
        for i,hit in enumerate(recovery_selected):
            eid=hit['evidence_id'];old=old_by_parent.get(eid)
            target=directory/f'evidence-{i+1}'
            proof=visual.verifier.verify(eid)
            if not proof['all_passed']:
                raise ValueError('Supplementary original source proof failed')
            _save(directory/f'source-proof-{i+1}.json',proof)
            prose_request={'native_unit_prose_recovery':True} if config.get('bounded_native_unit_prose_recovery',False) and proposal is None else {}
            binding=visual.prepare_region(eid,target,dpi=config['render_dpi'],
                readable_question=question,full_page_recovery=proposal is None,**prose_request)
            if old and (binding['source']['doc_sha256'],binding['source']['page'])!=(old['doc_sha256'],old['page']):
                raise ValueError('Recovery cannot change retrieved document or page')
            artifacts[eid]=target
            previous=old.get('readable_viewport',{}).get('native_word_records',[]) if old else []
            key=lambda w:tuple(w[k] for k in ('text','x0','x1','top','bottom'))
            old_words={key(w) for w in previous}
            new_words=binding['readable_viewport']['native_word_records']
            source=binding['source']
            changed=(old['image']['rgb_pixels_sha256']!=binding['selected_image']['rgb_pixels_sha256']
                     if old else all(s['image']['rgb_pixels_sha256']!=binding['selected_image']['rgb_pixels_sha256']
                         for s in old_by_parent.values()))
            changes.append({'parent_evidence_id':eid,'document_name':source['document_name'],
                'doc_sha256':source['doc_sha256'],'page':source['page'],
                'first_source_image_sha256':old['image']['sha256'] if old else None,
                'recovery_source_image_sha256':binding['selected_image']['sha256'],
                'actual_pixels_changed':changed,
                'original_viewport':old.get('readable_viewport',{}).get('viewport_bbox_top_left_points') if old else None,
                'recovery_viewport':binding['readable_viewport'].get('viewport_bbox_top_left_points'),
                'recovery_image_bbox_pixels':[0,0,*binding['selected_image']['size_pixels']],
                'recovery_geometry':binding['geometry'],
                'actual_added_native_words':[w for w in new_words if key(w) not in old_words],
                'native_paint_filter':binding['readable_viewport'].get('native_paint_filter'),
                'native_paint_filter_applied':binding['readable_viewport'].get('native_paint_filter') is not None})
            prose=binding['readable_viewport'].get('native_unit_prose_plan')
            if prose is not None:
                changes[-1]['actual_native_unit_prose_plan']=prose
                if not continuation_added and proposal is None:
                    from .native_unit_prose import continuation_source
                    pages={(s['doc_sha256'],s['page']) for s in recovery_selected}
                    following=continuation_source(question,prose,hit,visual.browser,pages) if len(pages)<3 else None
                    if following is not None:
                        additional=following['canonical_source']
                        recovery_selected.append(additional);recovery_hits.append(additional)
                        continuation_added=True
                        report.update(retrieval_source_pages_changed=True,
                            bounded_unit_continuation_acquisition=following,
                            source_continuation_semantics_verified=False,maximum_added_continuation_pages=1)
        report['source_changes']=changes
        if not proposal:
            report.update(recovery_route='same_original_page',
                reason=f'{kind} with cropped original source; add omitted original page once; validate a separate candidate')
            if continuation_added:
                report['recovery_route']='same_original_pages_plus_one_unit_continuation_page'
                report['reason']+='; actual selected unit prose ends without terminal punctuation; add same-document next verified physical page once, continuity unverified'
        if not any(c['actual_pixels_changed'] for c in changes):
            report['status']='no_new_pixels_no_reader_call'
            return original,prepared,report
        recovered=prepare_reader_visual_inputs(question,recovery_hits,visual=visual,
            primary_artifacts=artifacts,tokenizer=reader.context_tokenizer,budget=budget,
            max_new_tokens=config['max_new_tokens'],prompt_projection=config['prompt_projection'])
        _save(directory/'prepared-context.json',{'messages':recovered.messages,
              'image_catalog':recovered.citation_catalog,'source_preparation':recovered.report})
        native=prepare_native_visual_inputs(recovered,processor,budget=budget,
                                           max_new_tokens=config['max_new_tokens'])
        _save(directory/'input.json',{'messages':recovered.messages,'image_catalog':recovered.citation_catalog,
              'native_preprocessing':native.report,'input_token_ids':native.tensors['input_ids'][0].tolist(),
              'image_grid_thw':native.tensors['image_grid_thw'].tolist()})
        check=native.report['budget_validation']
        report['native_budget']=check
        if not check.get('token_budget_verified') or not check.get('tensor_contract_validated'):
            report['status']='blocked_native_budget_no_reader_call'
            return original,prepared,report
        report['actual_reader_calls']=1
        candidate=reader.read_visual(native,budget=budget,max_new_tokens=config['max_new_tokens'],
            max_seconds=config['generation_seconds'],
            full_source_citations=config.get('visual_citation_policy')=='full_selected_image_after_raw_binding_v1',
            compact_source_citations=config.get('compact_source_citations',False),
            answer_source_ids=config.get('answer_source_ids',False))
        _save(directory/'reader-output.json',candidate)
        from .source_calculation import compose_source_calculated_candidate
        candidate=compose_source_calculated_candidate(question,candidate,recovered)
        if config.get('bounded_native_unit_reference_candidate',False):
            from .unit_source_reference import compose_unit_reference_candidate
            candidate=compose_unit_reference_candidate(question,candidate,recovered)
        _save(directory/'effective-candidate.json',candidate)
        report.update(status='recovery_reader_completed',recovery_candidate_used=True,
            recovery_raw_output_sha256=hashlib.sha256(candidate.get('raw_output_for_validation','').encode()).hexdigest(),
            recovery_raw_candidate=candidate.get('diagnostic_candidate_output',{}).get('model_declared_output'),
            effective_candidate_origin=candidate.get('diagnostic_candidate_output',{}).get('effective_candidate_origin','Qwen Reader'),
            recovery_candidate_present=candidate.get('answer_candidate_present',False))
        return candidate,recovered,report
    except Exception as exc:
        report.update(status='recovery_failed_original_refusal_retained',
            error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        return original,prepared,report
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        _save(directory/'report.json',report)
