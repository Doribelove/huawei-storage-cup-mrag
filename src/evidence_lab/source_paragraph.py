"""Physical paragraph windows and bounded lexical continuation acquisition.

Parsed prose supplies search anchors only. Actual native words and rectangles
remain the facts shown to the Reader; continuity never establishes an answer.
"""
import re
import unicodedata
from .native_unit_prose import native_lines,source_box_lines
from .readable_region import native_word_lines,valid_box
from .safe_data import evidence_output


def paragraph_window(question,source,words,width,height):
    if source.get('type')!='text' or not valid_box(source.get('bbox_points_estimate'),width,height):
        raise ValueError('Native paragraph needs an original text-region rectangle')
    lines=source_box_lines(native_lines(words,width,height) or [],source['bbox_points_estimate'],width,height)
    text='\n'.join(line['text'] for line in lines)
    if not 1<=len(lines)<=120 or not 1<=len(text)<=12000:
        raise ValueError('Native paragraph has no bounded whole native lines')
    indices=sorted({i for line in lines for i in line['native_word_indices']})
    selected=[{k:words[i][k] for k in ('text','x0','top','x1','bottom')} for i in indices]
    box=[min(w['x0'] for w in selected),min(w['top'] for w in selected),
         max(w['x1'] for w in selected),max(w['bottom'] for w in selected)]
    viewport=[max(0,box[0]-1),max(0,box[1]-1),min(width,box[2]+1),min(height,box[3]+1)]
    return {'status':'selected_native_pdf_window','policy':'whole_native_source_paragraph_v1',
        'question':question,'kind':'native_source_paragraph','native_paragraph_window':True,
        'full_page_recovery':False,'native_object_bbox_top_left_points':box,
        'viewport_bbox_top_left_points':viewport,'margin_pdf_points':1,'dpi':144,
        'native_paragraph_lines':lines,'native_word_records':selected,'native_visible_text':text,
        'native_context_text':text,'native_spatial_word_lines':native_word_lines(selected),
        'quantized_source_region_used_for_word_selection':True,
        'source_bbox_precision':'Quantized acquisition rectangle; displayed full-line extent comes from actual native word coordinates',
        'actual_native_words_replace_parsed_quote_text':True,'association_verified':False,
        'spatial_word_groups_are_semantic_cells':False,'fact_localization_verified':False,
        'table_continuity_verified':False,'reference_pages_used':False}


def lexical_tokens(text):
    text=unicodedata.normalize('NFKC',text).replace('\u00ad','')
    def math_typography(match):
        value=match.group(1)
        # Acquisition anchors only: remove TeX font markup and spaces inside
        # its literal glyph runs. Source quotations/pixels are never changed.
        for _ in range(3):
            value=re.sub(r'\\(?:mathrm|mathbf|mathit|mathsf|mathtt|text|textrm)\s*\{([^{}]*)\}',
                lambda m:re.sub(r'\s+','',m.group(1)),value)
        return re.sub(r'\\[A-Za-z]+',' ',value).replace('{','').replace('}','')
    text=re.sub(r'\$([^$]*)\$',math_typography,text)
    text=re.sub(r'(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])','',text)
    return re.findall(r'[a-z]+',text.lower())


def anchor_span(canonical,native):
    """Unique actual boundary phrases; no numeric or semantic normalization."""
    tokens=lexical_tokens(native)
    if len(tokens)<16:return None
    positions=lambda phrase:[i for i in range(len(canonical)-len(phrase)+1) if canonical[i:i+len(phrase)]==phrase]
    first,last=positions(tokens[:8]),positions(tokens[-8:])
    if len(first)!=1 or len(last)!=1 or last[0]<first[0]:return None
    return {'start_token':first[0],'end_token':last[0]+8,
        'actual_first_native_anchor':tokens[:8],'actual_last_native_anchor':tokens[-8:],
        'anchor_words_per_boundary':8,'entire_text_or_answer_semantics_verified':False}


def plan_paragraph_sources(question,proposal,browser):
    """At most3 following physical pages and3 paragraph source rectangles."""
    import pdfplumber
    from .native_paint import form_bound_page
    body=proposal['body'];canonical=lexical_tokens(body['text'])
    if not 16<=len(canonical)<=5000:raise ValueError('Focus paragraph search anchors exceed budget')
    path=browser.verified_pdf(body['document_name'])
    report={'schema':'bounded-native-paragraph-continuation-v1','maximum_following_pages_inspected':3,
        'maximum_paragraph_sources':3,'canonical_text_is_acquisition_hint_only':True,
        'source_semantics_verified':False,'reference_pages_used':False,'scanned_physical_pages':[],'blocks':[]}
    with pdfplumber.open(path) as pdf:
        def block(source):
            page=pdf.pages[source['page']-1]
            page,_,_=form_bound_page(page)
            window=paragraph_window(question,source,page.extract_words(),page.width,page.height)
            span=anchor_span(canonical,window['native_visible_text'])
            return window,span
        first,span=block(body)
        if not span or span['start_token']!=0:
            raise ValueError('Original focus paragraph does not start at unique actual native anchors')
        selected=[body];cursor=span['end_token']
        report['blocks'].append({'evidence_id':body['evidence_id'],'page':body['page'],**span,
            'native_text':first['native_visible_text'],'native_word_records':first['native_word_records']})
        for number in range(body['page']+1,min(body['page']+4,len(pdf.pages)+1)):
            if cursor==len(canonical):break
            report['scanned_physical_pages'].append(number)
            candidates=[]
            for source in browser.records:
                if source['document_name']!=body['document_name'] or source['page']!=number or source['type']!='text':continue
                if not source.get('bbox_points_estimate'):continue
                try:window,span=block(source)
                except ValueError:continue
                if span and span['start_token']==cursor and span['end_token']>cursor:
                    candidates.append((source,window,span))
            if not candidates:continue
            if len(candidates)!=1 or len(selected)>=3:
                raise ValueError('Ambiguous or over-budget physical paragraph continuation')
            source,window,span=candidates[0];selected.append(evidence_output(source));cursor=span['end_token']
            report['blocks'].append({'evidence_id':source['evidence_id'],'page':number,**span,
                'native_text':window['native_visible_text'],'native_word_records':window['native_word_records']})
    if cursor!=len(canonical):
        raise ValueError('Bounded physical paragraph windows do not reach canonical paragraph end')
    report.update(complete_boundary_anchor_coverage=True,canonical_anchor_token_count=len(canonical),
        paragraph_count=len(selected),figure_retained_in_reader=len(selected)<3,
        parsed_cross_page_text_will_not_be_used_as_page_local_quote=True)
    if len(selected)<3:selected.append(proposal['figure'])
    return selected,report
