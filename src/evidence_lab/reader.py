"""Local pinned Qwen adapter and exact quote gate. No fallback or remote inference."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
from weakref import WeakKeyDictionary
import psutil
from .safe_data import sha256_file
from .reader_context import (MODEL_ID, REVISION, TOKENIZER_FILES, PinnedReaderTokenizer,
    ReaderBudget, assemble_reader_context, clean_reader_rows, VisualReaderContext,
    assemble_visual_reader_context, visual_reader_context_verified, native_processor_messages)

WEIGHTS={
    'model-00001-of-00004.safetensors':(4902275944,'d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa'),
    'model-00002-of-00004.safetensors':(4915962496,'8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5'),
    'model-00003-of-00004.safetensors':(4999831048,'83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192'),
    'model-00004-of-00004.safetensors':(2716270024,'0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5')}
CONFIGS={
    **TOKENIZER_FILES,
    'model.safetensors.index.json':(67759,'5857ec62530eb589b1876034dd5386f1d98131a6'),
    'preprocessor_config.json':(390,'2ea84a437d448ff71b08df68fdd949d5cc4ebb64'),
    'video_preprocessor_config.json':(385,'3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe')}

# Registration is established only after the real fixed weights load cleanly.
_LOADED_READERS = WeakKeyDictionary()


class ReaderUnavailable(RuntimeError):
    pass


def prepare_reader_sources(source_rows:list[dict])->list[dict]:
    """Testable source-context boundary; does not load a model or establish support."""
    clean=clean_reader_rows(source_rows,max_candidates=13)
    if sum(len(r['text']) for r in clean)>12000:
        raise ValueError('Reader context exceeds explicit character budget; no silent truncation')
    return clean


def prepare_reader_visual_inputs(question: str, retrieved_hits: list[dict], **kwargs) -> VisualReaderContext:
    """Reader-facing preparation entry; no reader/model construction or processor calls."""
    return assemble_visual_reader_context(question, retrieved_hits, **kwargs)


class VisualOutputInvalid(ValueError):
    """Distinguish malformed output from an unbound source, without a semantic claim."""
    def __init__(self, stage: str, reason: str):
        self.stage = stage
        super().__init__(f'{stage}: {reason}')


def _visual_output_structure(raw_text: str, *, compact_citations: bool = False) -> dict:
    def fail(reason): raise VisualOutputInvalid('structure', reason)
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result: fail('Duplicate JSON keys are forbidden')
            result[key] = value
        return result
    if not isinstance(raw_text, str) or len(raw_text) > 16000: fail('Output budget exceeded')
    try:
        value = json.loads(raw_text, object_pairs_hook=unique_keys)
    except json.JSONDecodeError as exc:
        raise VisualOutputInvalid('structure', 'Invalid JSON') from exc
    if not isinstance(value, dict) or set(value) != {'status', 'answer', 'citations'}:
        fail('Use exactly status, answer and citations')
    if value['status'] not in ('answered', 'not_enough_evidence') or not isinstance(value['answer'], str) or len(value['answer']) > 4000:
        fail('Invalid status/answer')
    citations = value['citations']
    if not isinstance(citations, list) or len(citations) > 8: fail('Citation count exceeded')
    if value['status'] == 'not_enough_evidence':
        if value['answer'] or citations: fail('Refusal must have empty answer/citations')
        return value
    if not value['answer'].strip() or not citations: fail('An answer candidate requires citations')
    for citation in citations:
        if not isinstance(citation, dict): fail('Invalid citation object')
        if compact_citations and set(citation) == {'image_id', 'observation'}:
            if not isinstance(citation['image_id'], str): fail('Invalid image ID')
            observation = citation['observation']
            if not isinstance(observation, str) or not observation.strip() or len(observation) > 1000:
                fail('Empty or over-budget observation')
            continue
        kind = citation.get('type')
        keys = {'type', 'image_id', 'quote'} if kind == 'text_quote' else {'type', 'image_id', 'region', 'observation'}
        if kind not in ('text_quote', 'visual_observation') or set(citation) != keys:
            fail('Use a typed citation; roles and support claims cannot be supplied by the model')
        if not isinstance(citation['image_id'], str): fail('Invalid image ID')
        declaration = citation['quote'] if kind == 'text_quote' else citation['observation']
        if not isinstance(declaration, str) or not declaration.strip() or len(declaration) > 1000:
            fail('Empty or over-budget quote/observation')
        if kind == 'visual_observation':
            region = citation['region']
            if (not isinstance(region, dict) or set(region) != {'coordinate_system', 'bbox_half_open'}
                    or region['coordinate_system'] != 'selected_image_pixels_top_left'):
                fail('Declare a region in the supplied image pixel coordinates')
            box = region['bbox_half_open']
            if not isinstance(box, list) or len(box) != 4 or any(type(n) is not int for n in box):
                fail('Visual region needs four integer pixel boundaries')
    return value


def validate_visual_reader_output(raw_text: str, prepared: VisualReaderContext) -> dict:
    """Validate structure/source binding; withhold every semantically unverified candidate."""
    # Citation mapping needs affine arithmetic, not optional PDF-rendering packages.
    def transform(matrix, point):
        a,b,c,d,e,f = matrix; x,y = point
        return [a*x+c*y+e,b*x+d*y+f]
    if not visual_reader_context_verified(prepared):
        raise ReaderUnavailable('Visual source packet is unregistered, changed or stale')
    value = _visual_output_structure(raw_text)
    seen = set(); resolved = []
    for citation in value['citations']:
        image_id = citation['image_id']; signature = json.dumps(citation, sort_keys=True)
        if image_id not in prepared.citation_catalog or signature in seen:
            raise VisualOutputInvalid('source_binding', 'Unknown image or duplicate citation')
        image = prepared.citation_catalog[image_id]
        seen.add(signature)
        resolved.append({key: image[key] for key in ('image_id', 'role', 'parent_evidence_id',
            'primary_retrieval_rank', 'doc_sha256', 'document_name', 'page', 'page_idx', 'pdf_member',
            'region_evidence_id', 'artifact_region_evidence_id', 'candidate_native_evidence_id', 'candidate_native_bbox_top_left_points',
            'source_word_union_bboxes_top_left_points', 'binding_id', 'table_context_id', 'image', 'pdf_to_image_pixel_matrix',
            'image_pixel_to_pdf_matrix', 'citation_support', 'table_continuity_verified', 'header_assigned_to_primary_table')})
        binding = resolved[-1]; binding['type'] = citation['type']
        if citation['type'] == 'text_quote':
            if citation['quote'] not in image['source_text_for_exact_quote']:
                raise VisualOutputInvalid('source_binding', 'Quote is not original source text')
            binding.update(quote=citation['quote'], exact_source_quote_bound=True)
        else:
            x0, y0, x1, y1 = citation['region']['bbox_half_open']; width, height = image['image']['size_pixels']
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise VisualOutputInvalid('source_binding', 'Declared pixel region is outside the bound image or empty')
            binding.update(observation=citation['observation'], observation_validated=False,
                observation_region={**citation['region'], 'selection_provenance': 'model_declared_pixel_region',
                    'pdf_polygon_bottom_left_canvas_points': [transform(image['image_pixel_to_pdf_matrix'], p)
                        for p in ((x0,y0),(x1,y0),(x1,y1),(x0,y1))],
                    'fact_localization_verified': False})
    candidate = value['status'] == 'answered'
    primary_cited = any(item['role'] == 'primary_retrieved_region' for item in resolved)
    # A syntactically valid, bound candidate can still contain arbitrary false prose.
    # No text filter or source hash is a semantic verifier. Final-facing fields stay empty.
    return {'status': 'not_enough_evidence', 'answer': '', 'citations': [],
            'output_contract_version': 2,
            'structure_validation': {'status': 'passed'},
            'source_binding_validation': {'status': 'passed', 'citations_bound': len(resolved),
                'primary_cited': primary_cited, 'binding_proves': 'source_identity_and_declared_location_only'},
            'evidence_support': {'status': 'unverified' if candidate else 'no_answer_candidate',
                'semantic_verifier_run': False, 'answer_entailment_verified': False,
                'visual_observations_verified': False, 'table_continuity_verified': False,
                'header_assigned_to_primary_table': False,
                'supplementary_context_confers_table_or_header_support': False},
            'presentation_status': 'withheld_unverified_candidate' if candidate else 'reader_refused',
            'diagnostic_candidate_output': {'not_for_answer_presentation': True,
                'model_declared_output': value, 'citation_bindings': resolved},
            'answer_candidate_present': candidate, 'supported_answer': False,
            'generation_provenance': 'not_established_by_output_validation',
            'semantic_correctness': 'not_established_by_structure_image_region_or_quote_binding'}


def reader_preflight(model_dir:Path,device:str='cpu')->dict:
    if device not in ('cpu','cuda'): raise ValueError('Reader device must be explicitly cpu/cuda')
    missing=[name for name in (*WEIGHTS,*CONFIGS) if not (model_dir/name).is_file()]
    available=psutil.virtual_memory().available
    reasons=[]
    if missing: reasons.append('Pinned local reader files are not present; no automatic download')
    if available<24*1024**3: reasons.append('Estimated24GiB host-memory headroom is unavailable')
    gpu_free=None
    if device=='cuda':
        import torch
        if not torch.cuda.is_available(): reasons.append('CUDA PyTorch runtime is unavailable')
        else:
            gpu_free=torch.cuda.mem_get_info()[0]
            if gpu_free<22*1024**3: reasons.append('Estimated22GiB full-precision CUDA headroom is unavailable')
    return {'model_id':MODEL_ID,'revision':REVISION,'device':device,'available_host_ram_bytes':available,
            'gpu_free_bytes':gpu_free,'estimated_required_host_headroom_bytes':24*1024**3,
            'weight_file_bytes':sum(size for size,digest in WEIGHTS.values()),
            'missing_files':missing,'ready':not reasons,'reasons':reasons,
            'note':'Memory values are conservative engineering estimates, not measured Qwen peaks; AGX connectivity is not assumed'}


def bind_full_source_citations(raw_text: str, prepared: VisualReaderContext) -> dict:
    """Explicit coarse citation policy after the original output passes all gates.

    Keeps the declared boxes for diagnostics; effective citations refer to the
    entire already-selected source image. This is not fact localization or
    semantic approval, and cannot rescue an illegal box, ID or quote.
    """
    import copy
    original=validate_visual_reader_output(raw_text,prepared)
    if not original['answer_candidate_present']:return original
    value=copy.deepcopy(original['diagnostic_candidate_output']['model_declared_output'])
    remapped=[]
    for citation in value['citations']:
        if citation['type']!='visual_observation':continue
        width,height=prepared.citation_catalog[citation['image_id']]['image']['size_pixels']
        before=list(citation['region']['bbox_half_open']);after=[0,0,width,height]
        citation['region']['bbox_half_open']=after
        remapped.append({'image_id':citation['image_id'],'model_declared_bbox_half_open':before,
            'effective_source_bbox_half_open':after,'selection_provenance':'full_already_selected_image_by_application_policy',
            'model_localization_verified':False,'same_retrieved_source_image':True})
    result=validate_visual_reader_output(json.dumps(value,ensure_ascii=False),prepared)
    result['diagnostic_candidate_output']['effective_candidate_output']=value
    result['diagnostic_candidate_output']['model_declared_output']=copy.deepcopy(
        original['diagnostic_candidate_output']['model_declared_output'])
    result['citation_region_policy']={'name':'full_selected_image_after_raw_binding_v1','remapped_regions':remapped,
        'raw_output_source_binding_validation':original['source_binding_validation'],
        'raw_declared_citation_bindings':original['diagnostic_candidate_output']['citation_bindings'],
        'precise_fact_localization_verified':False,'semantic_support_established_by_remapping':False}
    return result


def bind_compact_source_citations(raw_text: str, prepared: VisualReaderContext) -> dict:
    """Bind the explicit two-field citation contract to full supplied images.

    This opt-in contract asks the model only for a registered image ID and an
    observation. The application supplies type and full-image geometry from
    the live source packet, then runs the existing strict source gates. It
    never repairs JSON, changes an answer, or establishes semantic support.
    Already typed citations must still pass their original geometry gates.
    """
    import copy
    if not visual_reader_context_verified(prepared):
        raise ReaderUnavailable('Visual source packet is unregistered, changed or stale')
    value = _visual_output_structure(raw_text, compact_citations=True)
    projected = copy.deepcopy(value)
    seen = set(); derived = []
    for citation in projected['citations']:
        image_id = citation['image_id']
        if image_id not in prepared.citation_catalog or image_id in seen:
            raise VisualOutputInvalid('source_binding', 'Unknown image or duplicate citation')
        seen.add(image_id)
        if set(citation) != {'image_id', 'observation'}: continue
        width, height = prepared.citation_catalog[image_id]['image']['size_pixels']
        citation.update(type='visual_observation', region={
            'coordinate_system':'selected_image_pixels_top_left',
            'bbox_half_open':[0, 0, width, height]})
        derived.append({'image_id':image_id, 'effective_source_bbox_half_open':[0,0,width,height],
            'selection_provenance':'full_already_selected_image_by_application_contract',
            'model_declared_geometry':False, 'precise_fact_localization_verified':False})
    result = validate_visual_reader_output(json.dumps(projected, ensure_ascii=False), prepared)
    if result['answer_candidate_present']:
        result['diagnostic_candidate_output']['effective_candidate_output'] = projected
        result['diagnostic_candidate_output']['model_declared_output'] = value
        derived_ids = {item['image_id'] for item in derived}
        for binding in result['diagnostic_candidate_output']['citation_bindings']:
            if binding['image_id'] in derived_ids:
                binding['observation_region']['selection_provenance'] = 'full_already_selected_image_by_application_contract'
                binding['observation_region']['model_declared_geometry'] = False
        result['citation_region_policy'] = {
            'name':'application_full_image_only_for_compact_citations_v1',
            'derived_regions':derived, 'model_declared_geometry_preserved':True,
            'projected_output_source_binding_validation':result['source_binding_validation'],
            'precise_fact_localization_verified':False,
            'semantic_support_established_by_projection':False}
    result['compact_citation_projection'] = {
        'name':'image_id_observation_to_full_selected_image_v1', 'derived_regions':derived,
        'raw_output_sha256':hashlib.sha256(raw_text.encode('utf-8')).hexdigest(),
        'raw_model_output_preserved':True, 'answer_changed':False,
        'semantic_support_established_by_projection':False}
    return result


def bind_answer_source_ids(raw_text: str, prepared: VisualReaderContext) -> dict:
    """Opt-in answer/ID contract. Binding makes no semantic assertion.

    IDs are references, so repeating a known ID adds no fact and is projected
    to its first occurrence. Every original ID and raw byte remains recorded.
    Unknown IDs, malformed JSON and model-provided roles/flags still fail.
    """
    if not visual_reader_context_verified(prepared):
        raise ReaderUnavailable('Visual source packet is unregistered, changed or stale')
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value: raise VisualOutputInvalid('structure', 'Duplicate JSON keys')
            value[key] = item
        return value
    if not isinstance(raw_text, str) or len(raw_text) > 16000:
        raise VisualOutputInvalid('structure', 'Output budget exceeded')
    try:
        original = json.loads(raw_text, object_pairs_hook=unique)
    except json.JSONDecodeError as exc:
        raise VisualOutputInvalid('structure', 'Invalid JSON') from exc
    if not isinstance(original, dict) or set(original) != {'answer', 'evidence_ids'}:
        raise VisualOutputInvalid('structure', 'Use exactly answer and evidence_ids')
    answer, identifiers = original['answer'], original['evidence_ids']
    if (not isinstance(answer, str) or len(answer) > 4000
            or not isinstance(identifiers, list) or len(identifiers) > 8
            or any(not isinstance(item, str) for item in identifiers)):
        raise VisualOutputInvalid('structure', 'Invalid answer or source reference list')
    if (not answer and identifiers) or (answer and not answer.strip()) or (answer and not identifiers):
        raise VisualOutputInvalid('structure', 'Empty answer must have no evidence; nonempty answer needs evidence')
    if any(item not in prepared.citation_catalog for item in identifiers):
        raise VisualOutputInvalid('source_binding', 'Unknown source image ID')
    unique_ids = list(dict.fromkeys(identifiers))
    projected = {'status':'answered' if answer else 'not_enough_evidence', 'answer':answer, 'citations':[]}
    for image_id in unique_ids:
        source = prepared.citation_catalog[image_id]
        projected['citations'].append({'type':'visual_observation', 'image_id':image_id,
            'observation':'Application source reference; no model observation was requested.',
            'region':{'coordinate_system':'selected_image_pixels_top_left',
                      'bbox_half_open':[0,0,*source['image']['size_pixels']]}})
    result = validate_visual_reader_output(json.dumps(projected, ensure_ascii=False), prepared)
    diagnostic = result['diagnostic_candidate_output']
    diagnostic['model_declared_output'] = original
    diagnostic['effective_candidate_output'] = projected
    for binding in diagnostic['citation_bindings']:
        binding['observation_declared_by_model'] = False
        binding['observation_is_source_fact'] = False
        binding['observation_region']['selection_provenance'] = 'application_full_already_selected_source_reference'
        binding['observation_region']['model_declared_geometry'] = False
    result['answer_source_id_projection'] = {
        'contract':'answer_evidence_ids_v1', 'raw_output_sha256':hashlib.sha256(raw_text.encode()).hexdigest(),
        'raw_model_output_preserved':True, 'raw_identifiers':identifiers,
        'effective_identifiers':unique_ids, 'repeated_reference_count':len(identifiers)-len(unique_ids),
        'answer_changed':False, 'model_declared_observation':False,
        'semantic_support_established':False, 'fact_localization_verified':False}
    return result


def validate_reader_output(raw_text:str,source_rows:list[dict])->dict:
    """Exact quote binding does not establish answer correctness or semantic entailment."""
    if not isinstance(raw_text,str) or len(raw_text)>16000: raise ValueError('Reader output budget exceeded')
    value=json.loads(raw_text)
    if not isinstance(value,dict) or set(value)!={'status','answer','citations'}:
        raise ValueError('Reader must return the exact structured contract')
    if value['status'] not in ('answered','not_enough_evidence') or not isinstance(value['answer'],str) or len(value['answer'])>4000:
        raise ValueError('Invalid reader status/answer')
    citations=value['citations']
    if not isinstance(citations,list) or len(citations)>8: raise ValueError('Reader citation count exceeded')
    if value['status']=='not_enough_evidence':
        if value['answer'] or citations: raise ValueError('Refusal must not carry an unverified answer')
        return {**value,'quote_bindings_all_passed':True,'answer_generated':False,'supported_answer':False,
                'support_state':'reader_refused'}
    if not value['answer'].strip() or not citations: raise ValueError('Uncited generated answer is rejected')
    rows={row['evidence_id']:row for row in prepare_reader_sources(source_rows)}
    seen=set()
    for citation in citations:
        if not isinstance(citation,dict) or set(citation)!={'evidence_id','quote'}: raise ValueError('Invalid reader citation schema')
        eid=citation['evidence_id'];quote=citation['quote']
        if not isinstance(eid,str) or eid not in rows or eid in seen or not isinstance(quote,str) or not quote or len(quote)>1000 or quote not in rows[eid]['text']:
            raise ValueError('Unknown, duplicate or non-original reader quote')
        seen.add(eid)
    return {**value,'quote_bindings_all_passed':True,'answer_generated':True,'supported_answer':False,
            'support_state':'reader_generated_quote_bound_candidate',
            'semantic_correctness':'not_established_by_exact_quote_binding'}


def _generate_visual_batch(model, processor, batch: dict, *, device: str = 'cpu',
        budget: ReaderBudget = ReaderBudget(), generation_reserve: int = 256,
        max_new_tokens: int = 256, max_seconds: float = 60, require_visual: bool = True,
        _clock=time.perf_counter) -> dict:
    """Generation mechanics shared by the qualified live Reader and diagnostics.

    Callers verify the live model, processor and bound source inputs. Calling
    this adapter alone never establishes model provenance or answer support.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    if type(budget) is not ReaderBudget: raise ValueError('Use controlled generation budgets')
    budget.validate()
    if (type(generation_reserve) is not int or not 32 <= generation_reserve <= 512
            or type(max_new_tokens) is not int or not 32 <= max_new_tokens <= generation_reserve
            or isinstance(max_seconds,bool) or not isinstance(max_seconds,(int,float))
            or not 1 <= max_seconds <= 120 or device not in ('cpu','cuda')):
        raise ValueError('Visual generation request/reserve/deadline exceeded')
    required = {'input_ids','attention_mask'} | ({'pixel_values','image_grid_thw'} if require_visual else set())
    allowed = {'input_ids','attention_mask','pixel_values','image_grid_thw','mm_token_type_ids'}
    if (not required <= set(batch) <= allowed
            or ('pixel_values' in batch) != ('image_grid_thw' in batch)):
        raise ReaderUnavailable('Visual generation requires the complete native image tensor batch')
    if any(type(value) is not torch.Tensor for value in batch.values()):
        raise ReaderUnavailable('Visual generation needs tensors')
    ids = batch['input_ids']
    if (ids.dtype != torch.int64 or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1
            or ids.shape[1] > budget.max_prompt_tokens
            or ids.shape[1]+generation_reserve > budget.max_total_tokens):
        raise ReaderUnavailable('Actual expanded prompt/reserve budget exceeded')
    original = ids.detach().clone()
    started = _clock()
    inputs = {key:value.to(device) for key,value in batch.items()}
    class Deadline(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
            return _clock()-started >= max_seconds
    with torch.inference_mode():
        output = model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,
            return_dict_in_generate=False,stopping_criteria=StoppingCriteriaList([Deadline()]))
    elapsed = _clock()-started
    if elapsed > max_seconds:
        raise ReaderUnavailable('Visual generation exceeded deadline; candidate withheld')
    if (type(output) is not torch.Tensor or output.dtype != torch.int64 or output.ndim != 2
            or output.shape[0] != 1 or not torch.equal(output[0,:original.shape[1]].cpu(),original[0].cpu())):
        raise ReaderUnavailable('Generated tensor or original prompt prefix mismatch')
    tail = output[0,original.shape[1]:]
    if not 1 <= len(tail) <= max_new_tokens or torch.any(tail < 0).item():
        raise ReaderUnavailable('Generated token count/IDs violate the reserved budget')
    raw = processor.decode(tail,skip_special_tokens=True)
    if not isinstance(raw,str) or not raw.strip() or len(raw)>16000:
        raise ReaderUnavailable('Generated text is empty or exceeds output budget')
    return {'raw_output_for_validation':raw,'input_tokens':original.shape[1],
        'output_tokens':len(tail),'wall_ms':round(elapsed*1000,3),
        'input_token_ids':original[0].cpu().tolist(),'generated_token_ids':tail.cpu().tolist(),
        'hit_token_cap':len(tail)==max_new_tokens,
        'deadline_checked_between_tokens':True,'deadline_interrupts_running_kernels':False,
        'supported_answer':False}


class LocalQwenReader:
    """Runnable only after real local weights and sufficient hardware are verified."""
    def __init__(self,model_dir:Path,device:str='cpu',*,dtype:str='bfloat16'):
        if dtype not in ('float16','bfloat16') or (device=='cpu' and dtype=='float16'):
            raise ValueError('Use explicit supported non-quantized reader precision')
        preflight=reader_preflight(model_dir,device)
        if not preflight['ready']: raise ReaderUnavailable(json.dumps(preflight,ensure_ascii=False))
        for name,(size,digest) in WEIGHTS.items():
            path=model_dir/name
            if path.stat().st_size!=size or sha256_file(path)!=digest: raise ValueError('Pinned reader weight mismatch')
        for name,(size,digest) in CONFIGS.items():
            raw=(model_dir/name).read_bytes()
            if len(raw)!=size or hashlib.sha1(b'blob '+str(size).encode()+b'\0'+raw).hexdigest()!=digest:
                raise ValueError('Pinned reader processor/configuration mismatch')
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
        import torch
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        self.torch=torch;self.device=device
        self.context_tokenizer=PinnedReaderTokenizer(model_dir)
        self.processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
        self.model,loading=Qwen3VLForConditionalGeneration.from_pretrained(model_dir,local_files_only=True,
            trust_remote_code=False,use_safetensors=True,dtype=getattr(torch,dtype),output_loading_info=True,
            attn_implementation='eager')
        if any(loading[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Reader model state mismatch')
        self.model.to(device);self.model.eval();self.model.requires_grad_(False)
        self.identity={'model_id':MODEL_ID,'revision':REVISION,'device':device,'dtype':dtype,
                       'trust_remote_code':False,'local_files_only':True,'semantic_quality_verified':False}
        _LOADED_READERS[self] = (self.model,self.processor,device,dtype)

    def read(self,question:str,source_rows:list[dict],max_new_tokens:int=256,max_seconds:float=60,
             *,context_budget:ReaderBudget=ReaderBudget())->dict:
        if isinstance(source_rows, VisualReaderContext):
            raise ReaderUnavailable('Visual inputs cannot use the text-only token budget or reader route')
        # This first adapter is text-only; vision inputs and semantic checking require
        # a real hardware/model stage and must not be described as completed here.
        if (not isinstance(question,str) or not question.strip() or len(question)>500
            or isinstance(max_new_tokens,bool) or not isinstance(max_new_tokens,int) or not 32<=max_new_tokens<=512
            or isinstance(max_seconds,bool) or not isinstance(max_seconds,(int,float)) or not 1<=max_seconds<=120):
            raise ValueError('Reader question/source/generation budget exceeded')
        context=assemble_reader_context(question,source_rows,
            tokenizer=getattr(self,'context_tokenizer',None),budget=context_budget,max_new_tokens=max_new_tokens)
        if not context.report['ready_for_reader']:
            return {'status':'not_enough_evidence','answer':'','citations':[],
                'answer_generated':False,'supported_answer':False,'support_state':'reader_context_unavailable',
                'context_assembly':context.report,'actual_reader_run':False,'vision_inference_performed':False}
        clean=context.sources
        native_messages = native_processor_messages(context.messages)
        rendered = self.processor.apply_chat_template(native_messages,add_generation_prompt=True,tokenize=False)
        if not isinstance(rendered,str) or hashlib.sha256(rendered.encode('utf-8')).hexdigest() != context.encoding.rendered_sha256:
            raise ReaderUnavailable('Typed native content changed the full source prompt; generation refused')
        inputs=self.processor.apply_chat_template(native_messages,add_generation_prompt=True,tokenize=True,
            return_dict=True,return_tensors='pt',truncation=False)
        # A real processor must reproduce exactly the full prompt tokenized for assembly.
        if (inputs['input_ids'].shape[0]!=1 or
            tuple(inputs['input_ids'][0].tolist())!=context.encoding.token_ids):
            raise ReaderUnavailable('Reader processor/tokenizer full-prompt mismatch; generation refused')
        inputs={k:v.to(self.device) for k,v in inputs.items()};started=time.perf_counter()
        from transformers import StoppingCriteria,StoppingCriteriaList
        class Deadline(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs): return time.perf_counter()-started>=max_seconds
        with self.torch.inference_mode():
            generated=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,
                stopping_criteria=StoppingCriteriaList([Deadline()]))
        output_ids=generated[0,inputs['input_ids'].shape[-1]:]
        text=self.processor.decode(output_ids,skip_special_tokens=True)
        result=validate_reader_output(text,clean)
        return {**result,'reader_identity':self.identity,'context_assembly':context.report,
                'actual_reader_run':True,'input_tokens':inputs['input_ids'].shape[-1],
                'output_tokens':len(output_ids),'wall_ms':round((time.perf_counter()-started)*1000,3),
                'generation_deadline_enforced_between_tokens':True,'vision_inference_performed':False}

    def read_visual(self, prepared, *, budget:ReaderBudget=ReaderBudget(),
                    max_new_tokens:int=256,max_seconds:float=120,full_source_citations:bool=False,
                    compact_source_citations:bool=False, answer_source_ids:bool=False) -> dict:
        """Generate only from live bound native inputs and fixed loaded weights.

        Source binding remains separate from semantic support; candidates are
        withheld until a separate support verifier runs.
        """
        from .reader_processor import NativeVisualInputs, native_visual_inputs_verified
        if type(compact_source_citations) is not bool or (compact_source_citations and not full_source_citations):
            raise ValueError('Compact citations require the explicit full selected image policy')
        if (type(answer_source_ids) is not bool or
                (answer_source_ids and (not full_source_citations or compact_source_citations))):
            raise ValueError('Answer/ID contract requires its own explicit full selected source policy')
        if isinstance(prepared, NativeVisualInputs):
            if not native_visual_inputs_verified(prepared):
                raise ReaderUnavailable('Native visual source/processor/tensors/budget are unverified, changed or stale')
            state = _LOADED_READERS.get(self)
            if state is None or self.model is not state[0] or self.processor is not state[1] or self.device != state[2]:
                raise ReaderUnavailable('Real fixed-weight reader was not loaded or its live state changed')
            if max_new_tokens > prepared.report['budget_validation']['max_new_tokens_reserved']:
                raise ReaderUnavailable('Generation cannot exceed the native input reserve')
            generated = _generate_visual_batch(self.model,prepared.processor._backend,prepared.tensors,
                device=self.device,budget=budget,generation_reserve=max_new_tokens,
                max_new_tokens=max_new_tokens,max_seconds=max_seconds)
            if not native_visual_inputs_verified(prepared):
                raise ReaderUnavailable('Source/native tensor binding changed during generation')
            raw = generated['raw_output_for_validation']
            try:
                validated = (bind_answer_source_ids(raw,prepared.prepared) if answer_source_ids
                             else bind_compact_source_citations(raw,prepared.prepared) if compact_source_citations
                             else bind_full_source_citations(raw,prepared.prepared) if full_source_citations
                             else validate_visual_reader_output(raw,prepared.prepared))
            except VisualOutputInvalid as exc:
                validated = {'status':'not_enough_evidence','answer':'','citations':[],
                    'supported_answer':False,'answer_candidate_present':False,
                    'output_validation_error':{'stage':exc.stage,'reason':str(exc)}}
            return {**validated,**generated,'reader_identity':dict(self.identity),
                'actual_reader_run':True,'vision_inference_performed':True,
                'generation_provenance':'fixed_local_weights_native_visual_batch'}
        if not visual_reader_context_verified(prepared):
            raise ReaderUnavailable('Visual Reader source preparation is unverified, changed or stale')
        # Do not access processor/model, and never trust a caller-edited readiness boolean.
        raise ReaderUnavailable('Visual generation blocked: native multimodal processor, actual image_grid_thw, '
            'pixel tensors and expanded prompt tokens are unverified; placeholder/text tokens cannot authorize images')
