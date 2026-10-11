"""Literal numeric-unit source coverage; answer bytes and support stay separate.

This may add one already supplied adjacent source, never acquire a source or
assert that the numeric mentions belong to the requested entity. Normal fact
support must establish that interpretation from the whole actual source lines.
"""
import re
from .native_unit_prose import SOURCE_UNITS,requested_units
from .readable_region import valid_box


def native_unit_lines(source):
    viewport=source.get('readable_viewport',{});plan=viewport.get('native_unit_prose_plan')
    if not plan:return None
    if not viewport.get('full_page_recovery') or not viewport.get('native_unit_prose_recovery'):
        raise ValueError('Unit lines require actual full-page source recovery')
    words=viewport.get('native_word_records',[]);lines=plan.get('lines',[])
    if not 1<=len(words)<=5000 or not 1<=len(lines)<=200:
        raise ValueError('Unit source word/line budget')
    page_box=viewport.get('viewport_bbox_top_left_points',[])
    if not isinstance(page_box,list) or len(page_box)!=4 or page_box[:2]!=[0,0]:
        raise ValueError('Unit source must be an actual full native page')
    for line in lines:
        indices=line.get('native_word_indices',[])
        if (not isinstance(indices,list) or not indices or len(indices)>256
                or any(type(i) is not int or not 0<=i<len(words) for i in indices)
                or len(indices)!=len(set(indices))):
            raise ValueError('Invalid literal unit word indices')
        selected=[words[i] for i in indices]
        if any(not valid_box([w.get('x0'),w.get('top'),w.get('x1'),w.get('bottom')],page_box[2],page_box[3]) for w in selected):
            raise ValueError('Native unit words outside actual page')
        if line.get('text')!=' '.join(w['text'] for w in selected):
            raise ValueError('Generated unit quote is not actual native words')
        box=[min(w['x0'] for w in selected),min(w['top'] for w in selected),max(w['x1'] for w in selected),max(w['bottom'] for w in selected)]
        if line.get('bbox_top_left_points')!=box:
            raise ValueError('Literal unit coordinate mismatch')
    if sum(len(line['text']) for line in lines)>2400:
        raise ValueError('Whole literal source context budget')
    return lines


def unit_atoms(text,units):
    aliases={alias:unit for unit in units for alias in SOURCE_UNITS[unit]}
    if not aliases:return []
    pattern=r'(?<![\w.])([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*('+('|'.join(re.escape(s) for s in sorted(aliases,key=len,reverse=True)))+r')(?!\w)'
    result=[]
    for match in re.finditer(pattern,text):
        atom=(match.group(1),aliases[match.group(2)])
        if atom not in result:result.append(atom)
    return result


def requested_unit_reference_candidate(question,generated,catalog):
    if generated.get('answer_candidate_present') is not True:return None
    diagnostic=generated.get('diagnostic_candidate_output',{})
    candidate=diagnostic.get('effective_candidate_output',{})
    answer=candidate.get('answer');raw=diagnostic.get('model_declared_output',{})
    if not isinstance(answer,str) or not 1<=len(answer)<=2400 or raw.get('answer')!=answer:return None
    units=requested_units(question);atoms=unit_atoms(answer,units)
    if not 1<=len(atoms)<=8:return None
    citations=candidate.get('citations',[])
    ids=[c.get('image_id') for c in citations]
    if not ids or len(ids)>3 or len(set(ids))!=len(ids) or any(i not in catalog for i in ids):return None
    for citation in citations:
        source=catalog[citation['image_id']]
        if citation.get('region',{}).get('bbox_half_open')!=[0,0,*source['image']['size_pixels']]:return None
    witnesses={}
    for image_id,source in catalog.items():
        lines=native_unit_lines(source)
        if not lines:continue
        for line in lines:
            for atom in unit_atoms(line['text'],units):
                if atom in atoms:
                    witnesses.setdefault(atom,[]).append({'image_id':image_id,'document_name':source['document_name'],
                        'doc_sha256':source['doc_sha256'],'page':source['page'],
                        'literal_value':atom[0],'literal_unit':atom[1],'literal_source_line':line})
    if any(atom not in witnesses for atom in atoms):return None
    missing=[atom for atom in atoms if not any(w['image_id'] in ids for w in witnesses[atom])]
    if not missing:return None
    cited_with_lines=[catalog[i] for i in ids if native_unit_lines(catalog[i])]
    options=[]
    for image_id,source in catalog.items():
        if image_id in ids or source.get('role')!='primary_retrieved_region':continue
        if not any(source['doc_sha256']==s['doc_sha256'] and abs(source['page']-s['page'])==1 for s in cited_with_lines):continue
        if all(any(w['image_id']==image_id for w in witnesses[atom]) for atom in missing):options.append(image_id)
    if len(options)!=1:return None
    combined=ids+[options[0]]
    facts=[{'literal_value':atom[0],'literal_unit':atom[1],
            'actual_word_coordinate_witness':next(w for w in witnesses[atom] if w['image_id'] in combined)} for atom in atoms]
    return {'schema':'literal-unit-source-reference-candidate-v1','answer':answer,'evidence_ids':combined,
        'source_field_plan':{'unit_mentions':facts,'added_already_supplied_image_id':options[0],
            'maximum_added_source_references':1,'source_numeric_mentions_do_not_establish_metric_or_entity':True},
        'origin':'literal native unit word-coordinate source coverage; original answer unchanged',
        'answer_bytes_unchanged':True,'source_acquisition_performed':False,'semantic_support_established':False}


def compose_unit_reference_candidate(question,generated,prepared):
    from .reader_context import visual_reader_context_verified
    from .source_calculation import _compose_source_candidate
    if not visual_reader_context_verified(prepared):raise ValueError('Unit references require unchanged registered source input')
    candidate=requested_unit_reference_candidate(question,generated,prepared.citation_catalog)
    if candidate is None:return generated
    return _compose_source_candidate(generated,prepared,candidate,field='source_unit_reference_candidate',
        projection_origin='source unit word/coordinate reference coverage; raw answer and references preserved')


def unit_support_projection(citation,source):
    lines=native_unit_lines(source)
    if not lines:return None
    if citation.get('region',{}).get('bbox_half_open')!=[0,0,*source['image']['size_pixels']]:return None
    return {'native_unit_prose_lines':[line['text'] for line in lines],
        'literal_requested_units':source['readable_viewport']['native_unit_prose_plan']['requested_units'],
        'native_word_coordinate_witnesses_retained_in_source_records':True,
        'source_semantic_entity_metric_and_cross_page_association_verified':False}


def literal_unit_reason_conflicts(verdict,claims):
    """A true verdict cannot silently replace a fully enumerated unit list.

    Short/subset reasons or reasons with extra source context are not inferred
    to contradict a claim. This only flags the same-size concrete lists with
    differing values, never makes a rejected claim true or establishes facts.
    """
    from collections import Counter
    from decimal import Decimal
    source_claims={claim['id']:claim['claim'] for claim in claims}
    result=[]
    for decision in verdict['claims']:
        if decision['supported'] is not True:continue
        proposed=unit_atoms(source_claims[decision['id']],SOURCE_UNITS)
        explained=unit_atoms(decision['reason'],SOURCE_UNITS)
        if not 2<=len(proposed)<=8 or len(proposed)!=len(explained):continue
        if Counter(unit for _,unit in proposed)!=Counter(unit for _,unit in explained):continue
        normalize=lambda atoms:Counter((Decimal(number),unit) for number,unit in atoms)
        if normalize(proposed)==normalize(explained):continue
        result.append({'type':'true_flag_literal_unit_reason_mismatch','claim_id':decision['id'],
            'literal_claim_numeric_unit_mentions':[list(atom) for atom in proposed],
            'literal_reason_numeric_unit_mentions':[list(atom) for atom in explained],
            'source_facts_established':False,'action':'reject_this_verdict; at most existing one reread'})
    return result
