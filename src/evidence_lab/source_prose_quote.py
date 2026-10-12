"""Whole retrieved text-region candidates preserve source scope and frequency.

Only the bounded question-derived qualification acquisition route is eligible.
Literal bytes and their existing source bindings are rechecked separately from
semantic support. Parsed text is not declared independently verified PDF OCR.
"""
import hashlib
import re

from .bounded_selection import qualification_gap_request, qualification_search_cues, source_cue_forms, tokens
from .reader_context import visual_reader_context_verified


FIELD = 'source_qualification_quote_candidate'


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def requested_qualification_quote_candidate(question, catalog, acquired_evidence_id, request):
    if not isinstance(question, str) or not question.strip() or len(question)>500:
        return None
    if not isinstance(catalog, dict) or not 2<=len(catalog)<=3 or not isinstance(request, dict):
        return None
    if not isinstance(acquired_evidence_id, str) or not acquired_evidence_id:
        return None
    sources=[]
    for image_id, source in catalog.items():
        if (source.get('source_region_type')!='text' or source.get('role')!='primary_retrieved_region'
                or type(source.get('image_index')) is not int or source['image_index']<0):
            return None
        text=source.get('source_text_for_exact_quote')
        if not isinstance(text,str) or not text.strip() or len(text)>1600:
            return None
        if any(not isinstance(source.get(key),str) or not source[key] for key in
               ('doc_sha256','document_name','pdf_member','binding_id','parent_evidence_id')):
            return None
        if type(source.get('page')) is not int or source['page']<1:
            return None
        image=source.get('image',{})
        if not re.fullmatch('[0-9a-f]{64}',image.get('sha256','')):
            return None
        sources.append((image_id,source))
    sources.sort(key=lambda item:item[1]['image_index'])
    if (len({s['image_index'] for _,s in sources})!=len(sources)
            or len({s['parent_evidence_id'] for _,s in sources})!=len(sources)
            or len({s['doc_sha256'] for _,s in sources})!=1
            or len({s['page'] for _,s in sources})!=len(sources)):
        return None
    acquired=[(i,s) for i,s in sources if s['parent_evidence_id']==acquired_evidence_id]
    if len(acquired)!=1: return None
    prior=[{'doc_sha256':s['doc_sha256'],'text':s['source_text_for_exact_quote']}
           for _,s in sources if s['parent_evidence_id']!=acquired_evidence_id]
    if qualification_gap_request(question,prior)!=request:
        return None
    new_source=acquired[0][1];new_text=new_source['source_text_for_exact_quote']
    if (new_source['doc_sha256']!=request['document_sha256']
            or not qualification_search_cues(new_text,request['hazard'])
            or not all(source_cue_forms(a.casefold(),tokens(new_text)) for a in request['query_anchors'])):
        return None
    blocks=[];parts=[];offset=0
    for image_id,source in sources:
        original=source['source_text_for_exact_quote']
        left=len(original)-len(original.lstrip());right=len(original.rstrip())
        quote=original[left:right]
        # Retain the entire original region; never trim a qualifier or sentence.
        if parts: offset+=2
        blocks.append({'image_id':image_id,'parent_evidence_id':source['parent_evidence_id'],
            'binding_id':source['binding_id'],'document_name':source['document_name'],
            'doc_sha256':source['doc_sha256'],'pdf_member':source['pdf_member'],'page':source['page'],
            'original_image_sha256':source['image']['sha256'],
            'source_text_sha256':_sha(original),'source_character_span':[left,right],
            'answer_character_span':[offset,offset+len(quote)],'quote':quote,
            'whole_nonwhitespace_source_region_retained':True,
            'physical_fact_localization_verified':False})
        parts.append(quote);offset+=len(quote)
    answer='\n\n'.join(parts)
    if not 1<=len(answer)<=4000: return None
    return {'schema':'whole-source-qualification-quote-candidate-v1','answer':answer,
        'evidence_ids':[image_id for image_id,_ in sources],
        'origin':'application whole original retrieved text regions; not a Qwen answer or invented checklist',
        'semantic_support_established':False,
        'source_quote_plan':{'question_sha256':_sha(question),'acquired_evidence_id':acquired_evidence_id,
            'acquisition_request':request,'blocks':blocks,'separator':'\n\n',
            'binding_scope':'Exact canonical parsed source text and registered original PDF image/page; semantic meaning and OCR fidelity still need ordinary source verification',
            'parsed_text_is_independent_semantic_proof':False,'reference_labels_used':False}}


def compose_qualification_quote_candidate(question, generated, prepared, acquisition_report):
    if not visual_reader_context_verified(prepared):
        raise ValueError('Source quotation requires a live unchanged visual source context')
    acquired=(acquisition_report or {}).get('bounded_qualification_acquisition',{})
    if acquired.get('status')!='new_original_source_pending_verification':
        return generated
    candidate=requested_qualification_quote_candidate(question,prepared.citation_catalog,
        acquired.get('selected_evidence_id'),acquired.get('request'))
    if candidate is None:
        raise ValueError('Acquired qualification source cannot form a bounded whole-source quote candidate')
    from .source_calculation import _compose_source_candidate
    return _compose_source_candidate(generated,prepared,candidate,field=FIELD,
        projection_origin='Exact whole source-region quotation; original Reader bytes and failures retained')


def validate_qualification_quote_candidate(question, validated, catalog):
    supplied=validated.get(FIELD)
    if supplied is None: return None
    if not isinstance(supplied,dict) or not isinstance(supplied.get('source_quote_plan'),dict):
        raise ValueError('Malformed whole-source quote candidate')
    plan=supplied['source_quote_plan']
    expected=requested_qualification_quote_candidate(question,catalog,
        plan.get('acquired_evidence_id'),plan.get('acquisition_request'))
    if expected is None or any(supplied.get(key)!=value for key,value in expected.items()):
        raise ValueError('Whole-source quote bytes, question, scope or source binding changed')
    candidate=validated['diagnostic_candidate_output']['effective_candidate_output']
    ids=[c['image_id'] for c in candidate['citations']]
    if candidate['answer']!=expected['answer'] or ids!=expected['evidence_ids']:
        raise ValueError('Effective answer differs from independently reconstructed source quotation')
    return expected['source_quote_plan']


def quote_claim_groups(answer, plan, citations):
    positions={c['image_id']:i for i,c in enumerate(citations)}
    blocks=plan['blocks'];claims=[]
    if '\n\n'.join(b['quote'] for b in blocks)!=answer or not 1<=len(blocks)<=3:
        raise ValueError('Source quote groups must cover every actual answer byte')
    for i,block in enumerate(blocks):
        start,end=block['answer_character_span']
        if answer[start:end]!=block['quote'] or block['image_id'] not in positions:
            raise ValueError('Source quote claim is not an actual answer substring with its own source')
        claims.append({'id':i,'claim':answer[start:end],
            'scope_source_index':positions[block['image_id']],
            'kind':'whole_original_source_region_quote'})
    return claims
