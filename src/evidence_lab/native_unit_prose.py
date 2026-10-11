"""Literal native unit-line context and a bounded physical continuation hint.

Only source coordinates and words are selected. No values, inferred arithmetic,
answer phrases, document IDs, QA labels or semantic claims are generated.
"""
import math
import re
from .readable_region import tokens,valid_box

UNIT_PATTERNS={
    'nm':r'\b(?:nanomet(?:er|re)s?|nm)\b',
    'um':r'\bmicromet(?:er|re)s?\b|[µμ]m\b',
    'mm':r'\b(?:millimet(?:er|re)s?|mm)\b',
    'cm':r'\b(?:centimet(?:er|re)s?|cm)\b',
    'V':r'\bvolts?\b|(?<!\w)V(?!\w)',
}
SOURCE_UNITS={'nm':('nm',),'um':('µm','μm','um'),'mm':('mm',),'cm':('cm',),'V':('V','volt','volts')}


def requested_units(question):
    if not isinstance(question,str) or not 1<=len(question)<=500:return []
    return [unit for unit,pattern in UNIT_PATTERNS.items()
            if re.search(pattern,question,0 if unit=='V' else re.I)]


def cue_words(text):
    words=tokens(text)
    return words|{w[:-1] for w in words if len(w)>3 and w.endswith('s')}


def native_lines(words,width,height):
    if not isinstance(words,list) or not 1<=len(words)<=5000:return None
    rows=[]
    for index,word in sorted(enumerate(words),key=lambda p:(p[1].get('top',0),p[1].get('x0',0))):
        if (not isinstance(word.get('text'),str) or not word['text'] or
                any(type(word.get(k)) not in (int,float) or not math.isfinite(word[k]) for k in ('x0','x1','top','bottom'))
                or not (0<=word['x0']<word['x1']<=width and 0<=word['top']<word['bottom']<=height)):
            return None
        # An isolated centered page number at the physical footer is not prose.
        if word['text'].isdigit() and word['top']>.94*height and abs((word['x0']+word['x1'])/2-width/2)<.05*width:
            continue
        if not rows or abs(word['top']-rows[-1][0])>2:rows.append([word['top'],[]])
        rows[-1][1].append((index,word))
    result=[]
    for _,row in rows:
        segments=[]
        for entry in sorted(row,key=lambda p:p[1]['x0']):
            if not segments or entry[1]['x0']-segments[-1][-1][1]['x1']>12:segments.append([])
            segments[-1].append(entry)
        for segment in segments:
            result.append({'text':' '.join(w['text'] for _,w in segment),
                'native_word_indices':[i for i,_ in segment],
                'bbox_top_left_points':[min(w['x0'] for _,w in segment),min(w['top'] for _,w in segment),
                    max(w['x1'] for _,w in segment),max(w['bottom'] for _,w in segment)]})
    return result


def unit_line_matches(text,unit):
    aliases='|'.join(re.escape(alias) for alias in SOURCE_UNITS[unit])
    return bool(re.search(r'(?<![\w.])[+-]?\d+(?:\.\d+)?\s*(?:'+aliases+r')(?!\w)',text))


def source_box_lines(lines,box,width,height):
    if not valid_box(box,width,height):return []
    return [line for line in lines if box[0]<=(line['bbox_top_left_points'][0]+line['bbox_top_left_points'][2])/2<=box[2]
            and box[1]<=(line['bbox_top_left_points'][1]+line['bbox_top_left_points'][3])/2<=box[3]]


def native_unit_prose_plan(question,words,width,height,*,source_box=None,source_transcription=None):
    units=requested_units(question)
    if not units:return None
    lines=native_lines(words,width,height)
    if not lines:return None
    # Repair a whole empty parsed paragraph from its native source rectangle.
    if source_transcription=='':
        literal=source_box_lines(lines,source_box,width,height)
        text=' '.join(line['text'] for line in literal)
        if (literal and len(text)<=2400 and cue_words(question)&cue_words(text)
                and any(unit_line_matches(line['text'],unit) for line in literal for unit in units)):
            return {'schema':'literal-native-unit-prose-window-v1','requested_units':units,
                'anchors':[],'lines':literal,'native_word_count':len(words),
                'omitted_native_lines':len(lines)-len(literal),'whole_native_lines_retained':True,
                'maximum_windows':1,'maximum_context_characters':2400,
                'empty_source_transcription_repaired_from_native_rectangle':True,
                'source_bbox_top_left_points':list(source_box),'source_bbox_association_semantics_verified':False,
                'physical_page_end_in_selected_context':False,'next_page_context_hint':False,
                'last_actual_body_line':None,'cross_page_continuation_verified':False,
                'answer_fact_or_arithmetic_generated':False}
    query=cue_words(question)-{'exact','scale','utilized','employing','deposited','according','different','changes'}
    ranked=[]
    for index,line in enumerate(lines):
        matched_units=[unit for unit in units if unit_line_matches(line['text'],unit)]
        common=query & cue_words(line['text'])
        if not matched_units or not common:continue
        box=line['bbox_top_left_points'];neighbors=[]
        for j,neighbor in enumerate(lines):
            other=neighbor['bbox_top_left_points']
            horizontal=max(0,min(box[2],other[2])-max(box[0],other[0]))
            if abs(other[1]-box[1])<=80 and horizontal>=.5*min(box[2]-box[0],other[2]-other[0]):
                neighbors.append(j)
        coverage=query & set().union(*(cue_words(lines[j]['text']) for j in neighbors))
        if len(coverage)<3:continue
        ranked.append((len(common)+.1*len(coverage),index,neighbors,matched_units))
    ranked.sort(key=lambda entry:(-entry[0],entry[1]))
    selected=[];anchors=[]
    for _,index,neighbors,units_found in ranked:
        if len(anchors)>=2:break
        if index in selected:continue
        trial=sorted(set(selected)|set(neighbors))
        # Keep the whole coordinate window or omit it; never trim numeric lines.
        if sum(len(lines[j]['text']) for j in trial)>2400:continue
        selected=trial;anchors.append({'line_index':index,'literal_units':units_found})
    if not selected:return None
    selected_lines=[lines[j] for j in selected]
    # Choose the last actual selected column line, excluding the physical footer.
    last=max(lines,key=lambda line:(line['bbox_top_left_points'][3],line['bbox_top_left_points'][0]))
    at_end=last in selected_lines and last['bbox_top_left_points'][3]>.85*height
    unfinished=at_end and not re.search(r'[.!?;:]\s*$',last['text'])
    return {'schema':'literal-native-unit-prose-window-v1','requested_units':units,
        'anchors':anchors,'lines':selected_lines,'native_word_count':len(words),
        'omitted_native_lines':len(lines)-len(selected),'whole_native_lines_retained':True,
        'maximum_windows':2,'maximum_context_characters':2400,
        'physical_page_end_in_selected_context':at_end,'next_page_context_hint':unfinished,
        'last_actual_body_line':last if at_end else None,
        'cross_page_continuation_verified':False,'answer_fact_or_arithmetic_generated':False}


def continuation_source(question,plan,source,browser,existing_pages):
    if not plan or not plan.get('next_page_context_hint'):return None
    page=source['page']+1;doc=browser.documents[source['document_name']]
    if page>doc['page_count'] or (source['doc_sha256'],page) in existing_pages:return None
    query=cue_words(question);tail=cue_words(plan['last_actual_body_line']['text'])
    import pdfplumber
    from .native_paint import form_bound_page
    path=browser.verified_pdf(source['document_name'])
    with pdfplumber.open(path) as pdf:
        native_page=pdf.pages[page-1]
        native_page,_,_=form_bound_page(native_page)
        width,height=native_page.width,native_page.height
        words=native_page.extract_words()
    lines=native_lines(words,width,height)
    if not lines:return None
    options=[]
    for row in browser.records:
        if row['document_name']!=source['document_name'] or row['page']!=page or row['type']!='text':continue
        box=row.get('bbox_points_estimate')
        literal=source_box_lines(lines,box,width,height)
        text=' '.join(line['text'] for line in literal)
        terms=cue_words(text);common=query & terms;bridge=tail & terms
        if (not literal or len(common)<1 or len(bridge)<2
                or not any(unit_line_matches(line['text'],unit) for line in literal for unit in plan['requested_units'])):continue
        options.append((box[1],box[0],row['evidence_id'],row,literal,common,bridge))
    if not options:return None
    options.sort(key=lambda entry:entry[:3])
    _,_,_,selected,literal,common,bridge=options[0]
    return {'canonical_source':selected,'acquisition_reason':{
        'same_verified_document_next_physical_page':True,'previous_page':source['page'],'new_page':page,
        'actual_unfinished_body_line':plan['last_actual_body_line'],
        'maximum_added_pages':1,'cross_page_continuation_verified':False,
        'actual_next_source_native_lines':literal,'next_source_native_word_count':len(words),
        'physical_source_order_before_later_qualifying_sections':True,
        'query_cues':sorted(common),'original_source_tail_cues':sorted(bridge),
        'source_search_uses_answer_or_reference_pages':False}}
