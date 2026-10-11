"""Bounded question/source lexical diagnostics. No answers/reference annotations."""
import re
from .readable_region import tokens


def required_terms(question):
    # Exact years, uppercase identifiers and mathematical parameter names are
    # source-search cues, never evidence that those facts are true.
    found=re.findall(r'\b(?:19|20)\d{2}\b|\b[A-Z][A-Z0-9-]{1,}\b|[\u0370-\u03ff]+\d*',question)
    # Missing explicit performance wording is a source-gap search cue, not a
    # conclusion about the product. It uses the existing one pool expansion.
    found.extend(re.findall(r'\b(?:earthquake|seismic|fire|water)[ -](safe|resistant|proof)\b',question,re.I))
    return set(item.casefold() for item in found)


def select_question_pages(question,hits,count=2):
    if count not in (1,2):raise ValueError('Use bounded two-page selection')
    if not hits:return [],{'possible_missing_fields':True,'reason':'no_retrieved_source'}
    query=tokens(question);mandatory=required_terms(question)
    selected=[hits[0]];first=hits[0];pages={(first['doc_sha256'],first['page'])}
    original_order={item['evidence_id']:i for i,item in enumerate(hits)}
    options=[]
    first_words=tokens(first['text'])
    for item in hits[1:]:
        if (item['doc_sha256'],item['page']) in pages:continue
        words=tokens(item['text']);new_terms=(query-first_words)&words
        matching=mandatory&words
        coherent=item['doc_sha256']==first['doc_sha256']
        # Keep ranking as the dominant base; prefer same-document supporting
        # pages over unrelated filler when their source fields overlap the query.
        rank=original_order[item['evidence_id']]
        score=1/(rank+1)+.12*len(new_terms)+.25*len(matching)+(.60 if coherent else 0)
        options.append((score,rank,item,sorted(new_terms),coherent))
    options.sort(key=lambda item:(-item[0],item[1]))
    if count==2 and options:selected.append(options[0][2])
    visible=set().union(*(tokens(item['text']) for item in selected))
    missing=sorted(mandatory-visible)
    report={'policy':'question_only_two_page_source_coherence_v1','primary_original_rank':1,
        'selected_evidence_ids':[item['evidence_id'] for item in selected],
        'missing_exact_source_search_cues':missing,'possible_missing_fields':bool(missing),
        'heuristic_not_semantic_coverage':True,'reference_labels_used':False,
        'alternatives':[{'evidence_id':item[2]['evidence_id'],'original_rank':item[1]+1,
                         'score':item[0],'new_question_terms':item[3],'same_primary_document':item[4]}
                        for item in options],
        'retrieval_ranking_modified':False,'source_records_modified':False}
    return selected,report
