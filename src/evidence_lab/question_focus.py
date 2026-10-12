"""One bounded reread of a literal question-focus body/figure pair.

Question words and explicit original figure references propose sources only.
They do not authorize answers or alter the normal support verifier.
"""
import copy
import hashlib
import json
import re
import time
import traceback
from pathlib import Path

from .referenced_figure import _REFERENCE, caption_number
from .safe_data import evidence_output

_CONDITION = re.compile(r'\bunder\s+((?:[A-Za-z]+[ -]+){0,5}[A-Za-z]+)\s+conditions?\b', re.I)
_MODIFIERS = {'the','a','an','various','varying','different','changing','specific','experimental','tested'}
_COMMON = _MODIFIERS | {'what','which','how','are','and','for','from','with','under','condition','conditions',
    'does','these','those','their','this','that','into','the','article','detailed','impact','impacted'}


def condition_focus(question):
    if not isinstance(question,str) or not 1<=len(question)<=500:
        return None
    matches=list(_CONDITION.finditer(question))
    if len(matches)!=1:
        return None
    match=matches[0]
    terms=[w.lower() for w in re.findall(r'[A-Za-z]+',match.group(1)) if w.lower() not in _MODIFIERS]
    if not 1<=len(terms)<=3 or len(set(terms))!=len(terms):
        return None
    return {'literal_question_phrase':match.group(0),'question_character_span':list(match.span()),'terms':terms}


def _variants(term):
    # Only conventional whole-word suffixes; never arbitrary prefix matching.
    root=term[:-1] if len(term)>4 and term.endswith('s') and not term.endswith('ss') else term
    return {term,root,root+'s',root+'d' if root.endswith('e') else root+'ed',
            (root[:-1] if root.endswith('e') else root)+'ing'}


def _cue_spans(terms,text):
    result=[]
    for term in terms:
        pattern=r'\b(?:'+'|'.join(re.escape(x) for x in sorted(_variants(term),key=len,reverse=True))+r')\b'
        matches=list(re.finditer(pattern,text,re.I))
        if not matches:
            return []
        result.append({'question_term':term,'literal_source_matches':[
            {'text':m.group(0),'character_span':list(m.span())} for m in matches]})
    return result


def question_focus_sources(question,hits,selected,canonical_sources):
    focus=condition_focus(question)
    report={'schema':'literal-question-condition-source-proposal-v1','focus':focus,
        'maximum_body_regions':1,'maximum_figures':1,'new_search_queries':0,
        'source_candidates_from_current_retrieval_pool_only':True,'reference_pages_used':False,
        'proposal_establishes_answer_support':False,'eligible_pairs':[]}
    def decline(reason):
        report['reason']=reason
        return None,report
    if focus is None:
        return decline('No unique literal condition-focus phrase')
    if not 1<=len(hits)<=10 or not 1<=len(selected)<=3:
        return decline('Existing retrieval exceeds bounded source scope')
    if any(s['type']!='text' for s in selected):
        return decline('Existing nontext source retains priority')
    if any(_cue_spans(focus['terms'],s.get('text','')) for s in selected):
        return decline('Selected original text already contains the literal condition focus')
    primary=hits[0]['doc_sha256']
    if any(s['doc_sha256']!=primary for s in selected):
        return decline('Selected sources do not identify a single document')
    canonical={r['evidence_id']:r for r in canonical_sources if r['doc_sha256']==primary}
    query_words={w.lower() for w in re.findall(r'[A-Za-z]{3,}',question)}-_COMMON-set(focus['terms'])
    captions=[]
    present={s['evidence_id'] for s in selected}
    for hit in hits:
        if hit['doc_sha256']!=primary:
            continue
        source=canonical.get(hit['evidence_id'])
        if source is None or any(source[k]!=hit[k] for k in ('evidence_id','document_name','doc_sha256','page','type','text')):
            raise ValueError('Question-focus hit differs from canonical original source')
        if len(source['text'])>24000:
            raise ValueError('Question-focus source text exceeds bounded scope')
        number=caption_number(source['text']) if source['type']=='image' else None
        cues=_cue_spans(focus['terms'],source['text'])
        overlap=sorted(query_words & {w.lower() for w in re.findall(r'[A-Za-z]{3,}',source['text'])})
        if not number or not cues or len(overlap)<2 or source['evidence_id'] in present:
            continue
        duplicates=[r for r in canonical.values() if r['type']=='image' and caption_number(r['text'])==number]
        if len(duplicates)!=1:
            continue
        captions.append((source,number,cues,overlap))
    pairs=[]
    for caption,number,cues,overlap in captions:
        for hit in hits:
            if hit['doc_sha256']!=primary or hit['type']!='text' or hit['evidence_id'] in present:
                continue
            body=canonical[hit['evidence_id']]
            body_cues=_cue_spans(focus['terms'],body['text'])
            references=[m for m in _REFERENCE.finditer(body['text']) if m.group(1)==number]
            if not body_cues or not references:
                continue
            pairs.append({'route':'literal_question_condition_body_figure_v1','focus':copy.deepcopy(focus),
                'retrieved_evidence_ids':[h['evidence_id'] for h in hits],
                'original_selected_evidence_ids':[s['evidence_id'] for s in selected],
                'body':copy.deepcopy(evidence_output(body)),'figure':copy.deepcopy(evidence_output(caption)),
                'figure_number':number,'caption_focus_matches':cues,'body_focus_matches':body_cues,
                'additional_literal_question_caption_terms':overlap,
                'literal_body_figure_references':[{'text':m.group(0),'character_span':list(m.span())} for m in references],
                'answer_semantics_verified':False,'reference_pages_used':False})
    report['eligible_pairs']=[{'body':p['body']['evidence_id'],'figure':p['figure']['evidence_id']} for p in pairs]
    if len(pairs)!=1:
        return decline('No unique original body/figure pair for the missing question focus')
    report['reason']='Current sources miss a literal question condition; unique retrieved caption and explicitly linked original body contain it'
    return pairs[0],report


def _save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def prepare_focus_context(question,proposal,*,visual,tokenizer,processor,budget,config,output):
    from .reader import prepare_reader_visual_inputs
    from .reader_processor import prepare_native_visual_inputs
    directory=Path(output);directory.mkdir(parents=True,exist_ok=False)
    _save(directory/'source-proposal.json',proposal)
    from .source_paragraph import plan_paragraph_sources
    selected,continuity=plan_paragraph_sources(question,proposal,visual.browser)
    _save(directory/'physical-paragraph-selection.json',continuity)
    artifacts={};changes=[]
    if not continuity['figure_retained_in_reader']:
        # Preserve the actual figure/caption used for acquisition. Complete
        # prose takes the three-page Reader budget; no paragraph is truncated.
        anchor=directory/'acquisition-figure'
        anchor_proof=visual.verifier.verify(proposal['figure']['evidence_id'])
        if not anchor_proof['all_passed']:raise ValueError('Original acquisition figure proof failed')
        visual.prepare_region(proposal['figure']['evidence_id'],anchor,dpi=config['render_dpi'],
            readable_question=question,referenced_figure_context=True)
        if not visual.verify(anchor)['all_passed']:raise ValueError('Original acquisition figure image failed verification')
        _save(directory/'acquisition-figure-proof.json',anchor_proof)
    for i,source in enumerate(selected):
        eid=source['evidence_id'];proof=visual.verifier.verify(eid)
        if not proof['all_passed']:
            raise ValueError('Question-focus original source proof failed')
        _save(directory/f'source-proof-{i+1}.json',proof)
        target=directory/f'evidence-{i+1}'
        binding=visual.prepare_region(eid,target,dpi=config['render_dpi'],readable_question=question,
            native_paragraph_window=source['type']=='text',referenced_figure_context=source['type']=='image')
        if not visual.verify(target)['all_passed']:
            raise ValueError('Question-focus rendered source binding failed')
        artifacts[eid]=target
        changes.append({'evidence_id':eid,'document_name':source['document_name'],
            'doc_sha256':source['doc_sha256'],'page':source['page'],'source_type':source['type'],
            'original_text_sha256':hashlib.sha256(source['text'].encode()).hexdigest(),
            'image_sha256':binding['selected_image']['sha256'],
            'actual_original_native_words':binding['readable_viewport'].get('native_word_records',[]),
            'viewport_bbox_top_left_points':binding['readable_viewport'].get('viewport_bbox_top_left_points'),
            'original_source_new_to_reader':eid not in proposal['original_selected_evidence_ids'],
            'already_in_retrieval_pool':eid in proposal['retrieved_evidence_ids'],
            'native_paragraph_window':binding['readable_viewport'].get('native_paragraph_window',False)})
    prepared=prepare_reader_visual_inputs(question,selected,visual=visual,primary_artifacts=artifacts,
        tokenizer=tokenizer,budget=budget,max_new_tokens=config['max_new_tokens'],prompt_projection=config['prompt_projection'])
    native=prepare_native_visual_inputs(prepared,processor,budget=budget,max_new_tokens=config['max_new_tokens'])
    _save(directory/'input.json',{'messages':prepared.messages,'image_catalog':prepared.citation_catalog,
        'source_preparation':prepared.report,'native_preprocessing':native.report,
        'input_token_ids':native.tensors['input_ids'][0].tolist(),'image_grid_thw':native.tensors['image_grid_thw'].tolist(),
        'tensor_metadata':{k:{'shape':list(v.shape),'dtype':str(v.dtype),
            'sha256':hashlib.sha256(v.cpu().contiguous().numpy().tobytes()).hexdigest()} for k,v in native.tensors.items()}})
    return prepared,native,changes


def recover_question_focus(question,generated,prepared,*,hits,selected,visual,reader,processor,budget,config,output,prior_recovery):
    report={'schema':'bounded-question-focus-reread-v1','eligible':False,'attempt_limit':1,
        'shared_additional_reader_call_limit':1,'actual_reader_calls':0,'candidate_applied':False,
        'semantic_support_established':False,'reference_pages_used':False}
    if prior_recovery.get('actual_reader_calls',0):
        report['reason']='One additional Reader call already used by original recovery'
        return generated,prepared,report
    if not generated.get('answer_candidate_present'):
        report['reason']='Existing empty-answer recovery retains priority'
        return generated,prepared,report
    proposal,selection=question_focus_sources(question,hits,selected,visual.browser.records)
    report['source_selection']=selection
    if proposal is None:
        return generated,prepared,report
    report.update(eligible=True,source_proposal=proposal,
        reason='One original body/figure pair for the literal question condition absent from current selected text')
    started=time.perf_counter();directory=Path(output)
    try:
        recovered,native,changes=prepare_focus_context(question,proposal,visual=visual,
            tokenizer=reader.context_tokenizer,processor=processor,budget=budget,config=config,output=directory)
        report['actual_added_sources']=changes
        check=native.report['budget_validation'];report['native_budget']=check
        if not check.get('token_budget_verified') or not check.get('tensor_contract_validated'):
            report['status']='blocked_native_budget_original_candidate_retained'
            return generated,prepared,report
        report['actual_reader_calls']=1
        candidate=reader.read_visual(native,budget=budget,max_new_tokens=config['max_new_tokens'],
            max_seconds=config['generation_seconds'],
            full_source_citations=config.get('visual_citation_policy')=='full_selected_image_after_raw_binding_v1',
            compact_source_citations=config.get('compact_source_citations',False),
            answer_source_ids=config.get('answer_source_ids',False))
        _save(directory/'reader-output.json',candidate)
        if not candidate.get('answer_candidate_present'):
            report['status']='fresh_candidate_empty_or_invalid_original_candidate_retained'
            report['fresh_validation_error']=candidate.get('output_validation_error')
            return generated,prepared,report
        report.update(status='fresh_source_bound_candidate_requires_normal_support',candidate_applied=True,
            original_raw_output_sha256=hashlib.sha256(generated.get('raw_output_for_validation','').encode()).hexdigest(),
            fresh_raw_output_sha256=hashlib.sha256(candidate.get('raw_output_for_validation','').encode()).hexdigest())
        return candidate,recovered,report
    except Exception as exc:
        report.update(status='question_focus_failed_original_candidate_retained',
            error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        return generated,prepared,report
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        if directory.exists():_save(directory/'report.json',report)
