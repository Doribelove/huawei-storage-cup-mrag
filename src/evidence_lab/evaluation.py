"""Question-only runtime contract and explicitly named page-reference proxy metrics."""
from __future__ import annotations
import json
from pathlib import Path
import re
from .safe_data import reject_answer_fields


def load_questions(path: Path) -> list[dict]:
    if path.stat().st_size>2*1024**2: raise ValueError('Question file byte budget exceeded')
    payload=json.loads(path.read_text(encoding='utf-8'));reject_answer_fields(payload)
    if set(payload)!= {'version','questions'} or payload['version']!='question-only-v1':
        raise ValueError('Only the frozen question-only projection is accepted')
    questions=payload['questions']
    if not isinstance(questions,list) or not 1<=len(questions)<=1000: raise ValueError('Question count budget exceeded')
    seen=set()
    for item in questions:
        if set(item)!={'sample_id','question'} or not isinstance(item['sample_id'],str) or not item['sample_id'] or item['sample_id'] in seen:
            raise ValueError('Duplicate/invalid question record or answer/reference-field leak')
        if not isinstance(item['question'],str) or not 0<len(item['question'])<=500:
            raise ValueError('Question must have1..500characters; no implicit truncation')
        seen.add(item['sample_id'])
    return questions


def reference_pages(raw: dict, documents: dict) -> tuple[list[dict], dict]:
    """Scoring process only. Original paths are parsed, never opened."""
    pages=set();longdoc=set()
    pattern=re.compile(r'(?P<id>\d+)_page_(?P<page>\d+)\.(?:png|jpg|jpeg)$',re.I)
    def parse(path):
        if not isinstance(path,str): raise ValueError('Reference image path must be a string')
        match=pattern.search(path.replace('\\','/').rsplit('/',1)[-1])
        if not match: raise ValueError('Unsupported provided reference-page name')
        name=f"{raw['domain']}_{match['id']}";page=int(match['page'])
        if name not in documents or not 1<=page<=documents[name]['page_count']:
            raise ValueError('Reference page outside original PDF corpus')
        return name,page
    for path in raw['gt_image_paths']: pages.add(parse(path))
    for path in raw['longdoc_image_paths']: longdoc.add(parse(path))
    if not pages or not pages<=longdoc: raise ValueError('GT reference outside the provided document pages')
    names={name for name,page in longdoc}
    complete=all({page for doc,page in longdoc if doc==name}==set(range(1,documents[name]['page_count']+1)) for name in names)
    if not complete: raise ValueError('Cannot establish one-based complete original-document page alignment')
    return [{'document_name':name,'page':page} for name,page in sorted(pages)],{
        'one_based_full_document_page_alignment':True,'source_documents':sorted(names)}


def page_reference_metrics(predictions: list[dict], labels: list[dict], top_k: int = 5) -> dict:
    if not 1<=top_k<=100: raise ValueError('Invalid metric top_k')
    if len({p['sample_id'] for p in predictions})!=len(predictions) or len({l['sample_id'] for l in labels})!=len(labels):
        raise ValueError('Duplicate prediction or reference sample ID')
    byid={p['sample_id']:p for p in predictions}
    if set(byid)!={l['sample_id'] for l in labels}: raise ValueError('Prediction/reference coverage mismatch')
    routes=sorted({name for p in predictions for name in p['routes']})
    per_case=[]
    for label in labels:
        expected={(p['document_name'],p['page']) for p in label['reference_pages']}
        if not expected: raise ValueError('Empty page-reference label')
        result={}
        for route in routes:
            returned=byid[label['sample_id']]['routes'].get(route)
            if returned is None: raise ValueError('Missing prediction route')
            observed={(p['document_name'],p['page']) for p in returned[:top_k]}
            count=len(expected & observed)
            result[route]={'any_reference_page_hit':bool(count),'all_reference_pages_hit':count==len(expected),
                           'reference_page_coverage':count/len(expected),'matched_reference_pages':count,
                           'expected_reference_pages':len(expected)}
        per_case.append({'sample_id':label['sample_id'],'routes':result})
    summary={route:{'page_any_hit_at_k':sum(c['routes'][route]['any_reference_page_hit'] for c in per_case)/len(per_case),
                    'page_all_hit_at_k':sum(c['routes'][route]['all_reference_pages_hit'] for c in per_case)/len(per_case),
                    'macro_reference_page_coverage_at_k':sum(c['routes'][route]['reference_page_coverage'] for c in per_case)/len(per_case)} for route in routes}
    return {'sample_count':len(labels),'k':top_k,'metrics':summary,'per_case':per_case,
            'definition':'First k retrieved regions; unique original PDF pages matched to supplied GT image-page references',
            'limitations':['Page-reference proxy, not labeled region Recall, semantic relevance or answer accuracy',
                           'Provided30 samples are not an unseen competition test; future tuning makes them development data']}
