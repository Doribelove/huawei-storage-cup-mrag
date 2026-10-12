"""Whole actual native paragraph quotations preserve explicit qualifications.

Original PDF words/coordinates are independently re-extracted before support.
Exact source identity never substitutes for the ordinary semantic verifier.
"""
import copy
import hashlib
import json
from pathlib import Path

from .reader_context import ReaderBudget,visual_reader_context_verified
from .question_focus import condition_focus
from .safe_data import sha256_file
from .source_paragraph import paragraph_window

FIELD='source_native_paragraph_quote_candidate'
def _sha(text):return hashlib.sha256(text.encode()).hexdigest()


def requested_native_paragraph_quote(question,catalog,focus):
    if condition_focus(question)!=focus or focus is None:return None
    selected=[(i,s) for i,s in catalog.items() if s.get('readable_viewport',{}).get('native_paragraph_window')]
    if not 1<=len(selected)<=3:return None
    selected.sort(key=lambda x:x[1]['image_index'])
    if len({s['doc_sha256'] for _,s in selected})!=1 or len({s['page'] for _,s in selected})!=len(selected):return None
    if [s['page'] for _,s in selected]!=sorted(s['page'] for _,s in selected):return None
    if any(len({s[k] for _,s in selected})!=len(selected)
            for k in ('image_index','binding_id','parent_evidence_id')):return None
    blocks=[];parts=[];offset=0
    for image_id,source in selected:
        viewport=source['readable_viewport'];original=source.get('source_text_for_exact_quote')
        if (source.get('source_region_type')!='text' or source.get('role')!='primary_retrieved_region'
                or not isinstance(original,str) or not original.strip()
                or len(original.strip())>4000
                or original!=viewport.get('native_visible_text') or not viewport.get('native_word_records')
                or not source.get('local_original_pdf_path')
                or not source.get('native_paragraph_original_bbox_top_left_points')):
            return None
        left=len(original)-len(original.lstrip());right=len(original.rstrip());quote=original[left:right]
        if parts:offset+=2
        blocks.append({'image_id':image_id,'parent_evidence_id':source['parent_evidence_id'],
            'binding_id':source['binding_id'],'document_name':source['document_name'],'doc_sha256':source['doc_sha256'],
            'pdf_member':source['pdf_member'],'page':source['page'],'original_image_sha256':source['image']['sha256'],
            'source_text_sha256':_sha(original),'source_character_span':[left,right],
            'answer_character_span':[offset,offset+len(quote)],'quote':quote,
            'whole_nonwhitespace_source_region_retained':True,'physical_fact_localization_verified':False,
            'original_paragraph_bbox_top_left_points':source['native_paragraph_original_bbox_top_left_points']})
        parts.append(quote);offset+=len(quote)
    answer='\n\n'.join(parts)
    if len(answer)>ReaderBudget().max_characters:return None
    return {'schema':'complete-native-paragraph-quote-candidate-v1','answer':answer,
        'evidence_ids':[i for i,_ in selected],
        'origin':'application whole original native paragraph quotations; not a Qwen summary or invented inventory',
        'semantic_support_established':False,
        'source_quote_plan':{'question_sha256':_sha(question),'focus':focus,'blocks':blocks,'separator':'\n\n',
            'no_source_sentences_or_qualifiers_removed':True,'reference_labels_used':False,
            'binding_scope':'Whole actual native text windows and original PDF coordinates; font fidelity and answer meaning still require ordinary source verification'}}


def recheck_original_native_paragraphs(question,catalog,blocks):
    import pdfplumber
    from .native_paint import form_bound_page
    checks=[]
    for block in blocks:
        source=catalog[block['image_id']];path=Path(source['local_original_pdf_path'])
        if sha256_file(path)!=source['doc_sha256']:
            raise ValueError('Native paragraph original PDF hash changed')
        if sha256_file(Path(source['local_image_path']))!=source['image']['sha256']:
            raise ValueError('Native paragraph original image hash changed')
        with pdfplumber.open(path) as pdf:
            page,_,_=form_bound_page(pdf.pages[source['page']-1])
            original={'type':'text','bbox_points_estimate':source['native_paragraph_original_bbox_top_left_points']}
            expected=paragraph_window(question,original,page.extract_words(),page.width,page.height)
        viewport=source['readable_viewport']
        for key in ('native_visible_text','native_word_records','native_paragraph_lines',
                    'native_object_bbox_top_left_points','viewport_bbox_top_left_points'):
            if expected[key]!=viewport[key]:raise ValueError('Original PDF paragraph words or coordinates differ from quote source')
        if source['source_text_for_exact_quote']!=expected['native_visible_text']:
            raise ValueError('Page-local native quote differs from actual original PDF words')
        checks.append({'image_id':block['image_id'],'doc_sha256':source['doc_sha256'],'page':source['page'],
            'whole_native_text_sha256':_sha(expected['native_visible_text']),
            'native_words_reextracted':len(expected['native_word_records']),
            'paragraph_bbox_top_left_points':original['bbox_points_estimate'],
            'native_font_transcription_is_semantic_proof':False})
    return checks


def bind_native_paragraph_quote(question,candidate,prepared):
    """Bind each whole source block with the unchanged 4000-character parser.

    This application quotation contract may aggregate at most three separately
    bound original blocks. It is not model output and confers no semantic pass.
    """
    if not visual_reader_context_verified(prepared):
        raise ValueError('Native paragraph quotation needs live unchanged context')
    expected=requested_native_paragraph_quote(question,prepared.citation_catalog,
        candidate.get('source_quote_plan',{}).get('focus'))
    if expected is None or any(candidate.get(k)!=v for k,v in expected.items()):
        raise ValueError('Application quotation must contain all original native paragraph bytes')
    recheck_original_native_paragraphs(question,prepared.citation_catalog,expected['source_quote_plan']['blocks'])
    from .reader import bind_answer_source_ids
    parts=[bind_answer_source_ids(json.dumps({'answer':b['quote'],'evidence_ids':[b['image_id']]},
        ensure_ascii=False),prepared) for b in expected['source_quote_plan']['blocks']]
    bound=copy.deepcopy(parts[0]);diagnostic=bound['diagnostic_candidate_output']
    effective={'status':'answered','answer':expected['answer'],'citations':[
        citation for part in parts for citation in part['diagnostic_candidate_output']['effective_candidate_output']['citations']]}
    declared={'answer':expected['answer'],'evidence_ids':expected['evidence_ids']}
    diagnostic.update(model_declared_output=declared,effective_candidate_output=effective,
        citation_bindings=[binding for part in parts for binding in part['diagnostic_candidate_output']['citation_bindings']])
    bound['source_binding_validation']['citations_bound']=len(parts)
    bound['output_contract_version']='application_whole_native_paragraph_quote_v1'
    bound['application_quote_structure_validation']={
        'separately_bound_whole_original_blocks':len(parts),'block_answer_character_limit':4000,
        'aggregate_source_character_limit':ReaderBudget().max_characters,
        'aggregate_answer_characters':len(expected['answer']),'model_output_contract_unchanged':True,
        'source_identity_and_literal_bytes_only':True,'semantic_support_established':False}
    bound['answer_source_id_projection'].update(
        contract='application_whole_native_paragraph_quote_v1',
        raw_output_sha256=_sha(json.dumps(declared,ensure_ascii=False)),
        raw_identifiers=expected['evidence_ids'],effective_identifiers=expected['evidence_ids'],
        repeated_reference_count=0)
    assert bound['supported_answer'] is False and bound['answer']=='' and bound['citations']==[]
    return bound


def compose_native_paragraph_quote(question,generated,prepared,focus_report):
    if not (focus_report or {}).get('candidate_applied'):return generated
    if not visual_reader_context_verified(prepared):
        raise ValueError('Native quotation requires live unchanged source context')
    candidate=requested_native_paragraph_quote(question,prepared.citation_catalog,
        focus_report['source_proposal']['focus'])
    if candidate is None:raise ValueError('Complete acquired native paragraph cannot fit a whole source quotation')
    from .source_calculation import _compose_source_candidate
    return _compose_source_candidate(generated,prepared,candidate,field=FIELD,
        projection_origin='Exact whole native source paragraph; original first and supplementary Reader bytes separately preserved',
        native_quote_question=question)


def validate_native_paragraph_quote(question,validated,catalog):
    supplied=validated.get(FIELD)
    if supplied is None:return None
    if not isinstance(supplied,dict) or not isinstance(supplied.get('source_quote_plan'),dict):
        raise ValueError('Malformed native paragraph quote candidate')
    expected=requested_native_paragraph_quote(question,catalog,supplied['source_quote_plan'].get('focus'))
    if expected is None or any(supplied.get(k)!=v for k,v in expected.items()):
        raise ValueError('Whole native paragraph quote bytes, scope or source identity changed')
    effective=validated['diagnostic_candidate_output']['effective_candidate_output']
    if effective['answer']!=expected['answer'] or [c['image_id'] for c in effective['citations']]!=expected['evidence_ids']:
        raise ValueError('Effective answer is not the complete native source quotation')
    checks=recheck_original_native_paragraphs(question,catalog,expected['source_quote_plan']['blocks'])
    return {**expected['source_quote_plan'],'original_native_paragraphs_reextracted':checks}


def native_quote_source_declaration(index,citation,source,plan):
    blocks=[b for b in plan['blocks'] if b['image_id']==citation['image_id']]
    if len(blocks)!=1:raise ValueError('Native source quote declaration lacks one bound original block')
    block=blocks[0]
    return {'source_index':index,'image_id':citation['image_id'],'document_name':source['document_name'],
        'page':source['page'],'table_continuity_verified':False,'header_attribution_verified':False,
        'whole_native_quote_character_span':block['source_character_span'],
        'whole_native_text_sha256':block['source_text_sha256'],
        'native_paragraph_bbox_top_left_points':block['original_paragraph_bbox_top_left_points'],
        'duplicate_native_text_already_in_actual_claim':True,
        'actual_complete_original_source_pixels_follow':True,
        'exact_native_word_binding_is_answer_semantic_proof':False}
