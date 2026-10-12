"""Bounded question/source lexical diagnostics. No answers/reference annotations."""
import re
from .readable_region import tokens


def required_terms(question):
    # Exact years, uppercase identifiers and mathematical parameter names are
    # source-search cues, never evidence that those facts are true.
    found=re.findall(r'\b(?:19|20)\d{2}\b|\b[A-Z][A-Z0-9-]{1,}\b|[\u0370-\u03ff]+\d*',question)
    # Missing explicit performance wording is a source-gap search cue, not a
    # conclusion about the product. It uses the existing one pool expansion.
    for phrase in re.findall(r'\b(?:earthquake|seismic|fire|water)[ -](?:safe|resistant|proof)\b',question,re.I):
        found.extend(tokens(phrase))
    return set(item.casefold() for item in found)


def source_cue_forms(term, words):
    """Literal or regular plural search matches, never proof of a property."""
    forms = {term} & words
    if term.isascii() and term.isalpha() and len(term) >= 4 and not term.endswith('s'):
        forms |= {term + 's'} & words
    return sorted(forms)


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
        matching={term for term in mandatory if source_cue_forms(term,words)}
        coherent=item['doc_sha256']==first['doc_sha256']
        # Keep ranking as the dominant base; prefer same-document supporting
        # pages over unrelated filler when their source fields overlap the query.
        rank=original_order[item['evidence_id']]
        score=1/(rank+1)+.12*len(new_terms)+.25*len(matching)+(.60 if coherent else 0)
        options.append((score,rank,item,sorted(new_terms),coherent))
    options.sort(key=lambda item:(-item[0],item[1]))
    if count==2 and options:selected.append(options[0][2])
    visible=set().union(*(tokens(item['text']) for item in selected))
    missing=sorted(term for term in mandatory if not source_cue_forms(term,visible))
    report={'policy':'question_only_two_page_source_coherence_v1','primary_original_rank':1,
        'selected_evidence_ids':[item['evidence_id'] for item in selected],
        'missing_exact_source_search_cues':missing,'possible_missing_fields':bool(missing),
        'heuristic_not_semantic_coverage':True,'reference_labels_used':False,
        'alternatives':[{'evidence_id':item[2]['evidence_id'],'original_rank':item[1]+1,
                         'score':item[0],'new_question_terms':item[3],'same_primary_document':item[4]}
                        for item in options],
        'retrieval_ranking_modified':False,'source_records_modified':False,
        'cue_forms_are_search_only_not_fact_support':True,
        'selected_source_cue_forms':{item['evidence_id']:{term:source_cue_forms(term,tokens(item['text']))
            for term in sorted(mandatory) if source_cue_forms(term,tokens(item['text']))} for item in selected}}
    return selected,report


def qualification_search_cues(text, hazard):
    # Same linguistic field sought by the existing negative source audit. A
    # parsed phrase can nominate a PDF source, but cannot satisfy that audit:
    # original native text/pixels and normal semantic checks still run later.
    topic = r'(?:earthquake|seismic)' if hazard in ('earthquake','seismic') else re.escape(hazard)
    pattern = (topic + r'[ -](?:safe|resistan\w*|proof|safety|performance|design)\b|'
               r'\b(?:resistan\w*|safety|performance)\b[^.!?\n]{0,35}' + topic)
    return [match.group(0) for match in re.finditer(pattern, text, re.I)]


def qualification_gap_request(question, selected):
    if not selected or len({source['doc_sha256'] for source in selected}) != 1:
        return None
    phrases = re.findall(r'\b(earthquake|seismic|fire|water)[ -](safe|resistant|proof)\b', question, re.I)
    anchors = list(dict.fromkeys(re.findall(r'\b[A-Z]{2,12}\b', question)))
    if not phrases or not 1 <= len(anchors) <= 3:
        return None
    text = '\n'.join(source['text'] for source in selected)
    for hazard, property_name in phrases:
        hazard = hazard.casefold()
        if qualification_search_cues(text, hazard):
            continue
        return {'query': ' '.join(anchors + [hazard]), 'query_anchors': anchors,
                'hazard': hazard, 'requested_qualification': hazard + '-' + property_name.casefold(),
                'document_sha256': selected[0]['doc_sha256'], 'top_k': 5,
                'maximum_new_sources': 1, 'maximum_retrieval_calls': 1,
                'reason': 'Question names a material qualification missing from selected source strings',
                'search_cues_are_not_fact_evidence': True, 'reference_labels_used': False}
    return None


def supplementary_qualification_source(request, hits, selected):
    prior_ids = {source['evidence_id'] for source in selected}
    prior_pages = {(source['doc_sha256'],source['page']) for source in selected}
    for source in hits[:request['top_k']]:
        if (source['evidence_id'] in prior_ids or source['type'] != 'text'
                or source['doc_sha256'] != request['document_sha256']
                or (source['doc_sha256'],source['page']) in prior_pages):
            continue
        words = tokens(source['text'])
        if not all(source_cue_forms(anchor.casefold(),words) for anchor in request['query_anchors']):
            continue
        cues = qualification_search_cues(source['text'], request['hazard'])
        if cues:
            return source, cues
    return None, []


def acquire_question_sources(question, hits, backend, route, count=2, *, qualification_acquisition=False):
    """One bounded source-search; never retry a verdict or use answer labels."""
    if type(qualification_acquisition) is not bool:
        raise ValueError('Qualification source acquisition must be explicitly boolean')
    selected, initial = select_question_pages(question,hits,count)
    request = qualification_gap_request(question,selected) if qualification_acquisition else None
    if backend is not None and request is not None:
        extra, details = backend.search(request['query'],route,request['top_k'])
        source, cues = supplementary_qualification_source(request,extra,selected)
        report = {'initial':initial, 'supplementary_retrieval_count':1,
                  'possible_missing_fields':True, 'bounded_qualification_acquisition':{
                      'request':request, 'retrieval_details':details,
                      'returned_evidence_ids':[item['evidence_id'] for item in extra],
                      'selected_evidence_id':source['evidence_id'] if source else None,
                      'literal_search_cues':cues, 'source_binding_and_support_pending':True,
                      'max_new_sources':1, 'appended_source_order_is_acquisition_order':True,
                      'status':'new_original_source_pending_verification' if source else 'bounded_search_exhausted'},
                  'reference_labels_used':False, 'source_records_modified':False}
        if source is not None:
            # Preserve original ranking entries and both complementary pages.
            # The new source has a separate query lineage; no fused score or
            # claim of factual support is invented for it.
            selected = list(selected) + [source]
            if source['evidence_id'] not in {item['evidence_id'] for item in hits}:
                hits = list(hits) + [source]
        report['selected_evidence_ids'] = [item['evidence_id'] for item in selected]
        return hits, selected, report
    if backend is not None and initial['possible_missing_fields']:
        original_hits = hits
        hits, details = backend.search(question,route,10)
        selected, after = select_question_pages(question,hits,count)
        return hits, selected, {'initial':initial,'after_single_pool_expansion':after,
            'supplementary_retrieval_count':1,'initial_evidence_ids':[h['evidence_id'] for h in original_hits],
            'expanded_top_k':10,'additional_details':details,'possible_missing_fields':after['possible_missing_fields']}
    return hits, selected, initial
