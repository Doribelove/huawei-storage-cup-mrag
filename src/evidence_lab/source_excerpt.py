"""Bounded literal PDF word excerpts for omitted long native prose.

This selects source words, never creates claims or required answer lists. It is
only used for verification of an existing candidate inside a full cited image.
"""
import json
import math
from .readable_region import tokens


def long_prose_excerpt(source, question, candidate, citation_box, max_bytes=1800):
    viewport=source.get('readable_viewport',{})
    region_type=source.get('source_region_type')
    if not isinstance(region_type,str) or region_type.casefold()!='text':return None
    size=source.get('image',{}).get('size_pixels')
    if not size or list(citation_box)!=[0,0,*size]:return None
    words=viewport.get('native_word_records',[])
    if not isinstance(words,list) or not 1<=len(words)<=5000:return None
    query=tokens(question)|tokens(candidate)
    if not query:return None
    rows=[]
    for i,w in sorted(enumerate(words),key=lambda pair:(pair[1].get('top',0),pair[1].get('x0',0))):
        if (not isinstance(w.get('text'),str) or not w['text'] or
                any(type(w.get(k)) not in (int,float) or not math.isfinite(w[k])
                    for k in ('x0','top','x1','bottom'))):return None
        if not rows or abs(w['top']-rows[-1][0])>2:rows.append([w['top'],[]])
        rows[-1][1].append((i,w))
    lines=[]
    for _,row in rows:
        segments=[]
        for i,w in sorted(row,key=lambda pair:pair[1]['x0']):
            if not segments or w['x0']-segments[-1][-1][1]['x1']>20:segments.append([])
            segments[-1].append((i,w))
        for segment in segments:
            text=' '.join(w['text'] for _,w in segment)
            box=[min(w['x0'] for _,w in segment),min(w['top'] for _,w in segment),
                 max(w['x1'] for _,w in segment),max(w['bottom'] for _,w in segment)]
            lines.append({'text':text,'native_word_indices':[i for i,_ in segment],
                          'pdf_bbox_top_left_points':[round(v,3) for v in box]})
    ranked=sorted(range(len(lines)),key=lambda i:(-len(tokens(lines[i]['text'])&query),i))
    selected=[];anchors=[]
    for i in ranked:
        if len(anchors)>=2:break
        if not tokens(lines[i]['text'])&query or i in selected:continue
        anchor=lines[i]['pdf_bbox_top_left_points'];neighbors=[]
        for j,line in enumerate(lines):
            b=line['pdf_bbox_top_left_points']
            overlap=max(0,min(b[2],anchor[2])-max(b[0],anchor[0]))
            if abs(b[1]-anchor[1])<=30 and overlap>=.5*min(b[2]-b[0],anchor[2]-anchor[0]):
                neighbors.append(j)
        # Keep whole actual lines, preferring the matched line and its nearest
        # column neighbors; never truncate a line into a new source sentence.
        order=[i]+sorted((j for j in neighbors if j!=i),key=lambda j:(abs(lines[j]['pdf_bbox_top_left_points'][1]-anchor[1]),j))
        appended=False
        for j in order:
            if j in selected:continue
            trial=sorted(selected+[j])
            if len(json.dumps([lines[n] for n in trial],ensure_ascii=False,separators=(',',':')).encode())<=max_bytes:
                selected=trial;appended=True
        if appended:anchors.append(i)
    if not selected:return None
    return {'selection':'question_and_literal_candidate_terms_maximum_two_native_line_windows',
            'native_word_count':len(words),'total_native_lines':len(lines),'selected_native_lines':len(selected),
            'omitted_native_lines':len(lines)-len(selected),'spans':[lines[i] for i in selected],
            'transcription_only_not_semantic_proof':True,'outside_cited_image_text_used':False,
            'generated_answer_phrases_or_required_lists':False}
