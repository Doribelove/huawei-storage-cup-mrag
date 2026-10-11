"""Question-only selection of actual PDF table/image windows; no OCR or gold."""
from __future__ import annotations
import math
import re

STOP = frozenset('the a an of and or to for in on as is are was were be by with from '
                 'what which how do does did their its it this that these those between '
                 'according context outlined detailed utilized used measured compare'.split())


def tokens(text):
    return set(re.findall(r'\w+', str(text).casefold())) - STOP


def valid_box(box, width, height):
    return (isinstance(box, (list, tuple)) and len(box)==4
            and all(type(n) in (int,float) and math.isfinite(n) for n in box)
            and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def overlap(a, b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union else 0


def native_word_lines(words):
    """Retain real PDF words by visible baseline and horizontal whitespace.

    These are spatial word groups, never verified semantic table cells.
    """
    rows=[]
    for word in sorted(words,key=lambda w:(w['top'],w['x0'])):
        if not rows or abs(word['top']-rows[-1][0])>3:
            rows.append([word['top'],[word]])
        else:rows[-1][1].append(word)
    result=[]
    for _,row in rows:
        groups=[];last=None
        for word in sorted(row,key=lambda w:w['x0']):
            if last is None or word['x0']-last['x1']>10:groups.append(word['text'])
            else:groups[-1]+=' '+word['text']
            last=word
        result.append(groups)
    return result


def choose_window(question, source, candidates, width, height):
    """Pure selector. Candidate boxes must originate from actual PDF objects."""
    if not isinstance(question,str) or not question.strip() or len(question)>500:
        raise ValueError('Question-only viewport request exceeds budget')
    if not valid_box([0,0,width,height],width,height):
        raise ValueError('Invalid page geometry')
    query=tokens(question); source_words=tokens(source.get('text',''))
    table_question=source.get('type')=='table' or bool(query & {
        'table','tax','taxes','rate','rates','parameter','parameters','unit','units','timing',
        'profits','state','states','misclassification','cap','caps','min','max','dimensions'})
    ranked=[]
    for item in candidates:
        box=item['bbox']
        if not valid_box(box,width,height): raise ValueError('PDF object window outside page')
        area=(box[2]-box[0])*(box[3]-box[1])
        if not .015 <= area/(width*height) <= .90: continue
        words=tokens(item.get('text',''))
        common=sorted(query & words)
        score=3*len(common)+.05*len(source_words & words)
        if item['kind'] in ('native_table','source_region_native_word_union'):
            if not table_question: continue
            score+=1
            original=source.get('source_bbox')
            if (source.get('bbox_coordinate_system')=='PDF native points; top-left; pdfplumber'
                    and valid_box(original,width,height)):
                score+=8*overlap(original,box)
        elif item['kind']=='embedded_pdf_image':
            if not table_question and source.get('type')!='image': continue
            score+=area/(width*height)
        else: raise ValueError('Unknown native PDF viewport kind')
        ranked.append((score,item,common))
    if not ranked:
        return {'status':'full_page_fallback','reason':'no_bounded_native_table_or_image_window',
                'question':question,'association_verified':False}
    ranked.sort(key=lambda v:(-v[0],v[1]['bbox'][1],v[1]['bbox'][0]))
    score,item,common=ranked[0]
    box=item['bbox']; margin=28 if item['kind'] in ('native_table','source_region_native_word_union') else 18
    viewport=[max(0,box[0]-margin),max(0,box[1]-margin),
              min(width,box[2]+margin),min(height,box[3]+margin)]
    table_window=item['kind'] in ('native_table','source_region_native_word_union')
    if table_window:
        # Numeric-only parsed boxes do not delimit the table's row labels.
        # Keep the actual page width over the same bounded vertical band.
        viewport[0]=0;viewport[2]=width
    pixels144=math.ceil((viewport[2]-viewport[0])*2)*math.ceil((viewport[3]-viewport[1])*2)
    dpi=144 if pixels144<=1_350_000 else 100
    return {'status':'selected_native_pdf_window','policy':'question_only_pdf_window_v2',
            'question':question,'kind':item['kind'],'native_object_index':item['index'],
            'native_object_bbox_top_left_points':box,'viewport_bbox_top_left_points':viewport,
            'margin_pdf_points':margin,'matched_question_tokens':common,'selection_score':score,
            'candidate_count':len(ranked),'dpi':dpi,'association_verified':False,
            'full_width_row_label_context_retained':table_window,
            'table_continuity_verified':False,'fact_localization_verified':False,
            'source_cell_rows':item.get('cells'),'source_cell_extraction':'pdfplumber_native_uninterpreted_cells',
            'scope':'same already selected verified PDF page; source viewport only, no semantic proof'}


def plan_pdf_window(path, question, source, width, height, *, full_page_recovery=False,native_unit_prose_recovery=False):
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        page=pdf.pages[source['page']-1]
        if page.width!=width or page.height!=height:
            raise ValueError('PDF parser geometry disagrees on viewport page')
        if type(full_page_recovery) is not bool:
            raise ValueError('Full-page recovery flag must be boolean')
        if type(native_unit_prose_recovery) is not bool or (native_unit_prose_recovery and not full_page_recovery):
            raise ValueError('Unit prose recovery needs the existing full original-page recovery')
        paint_audit=None
        if full_page_recovery:
            from .native_paint import form_bound_page
            page,unfiltered,paint_audit=form_bound_page(page)
        words=page.extract_words()
        if len(words)>15000: raise ValueError('Native source word budget exceeded')
        def nearby(box, margin=28):
            return ' '.join(w['text'] for w in words if w['x0']>=box[0]-margin
                and w['x1']<=box[2]+margin and w['top']>=box[1]-margin and w['bottom']<=box[3]+margin)
        candidates=[];omitted=[]
        tables=page.find_tables()
        if len(tables)>64 or len(page.images)>64: raise ValueError('PDF object window budget exceeded')
        for index,table in enumerate(tables):
            cells=table.extract()
            if len(cells)>256 or sum(len(row) for row in cells)>2048: continue
            text=' '.join(str(cell or '') for row in cells for cell in row)
            if len(text)>12000: continue
            box=list(table.bbox)
            if valid_box(box,width,height):
                candidates.append({'kind':'native_table','index':index,'bbox':box,
                                   'text':text+' '+nearby(box),'cells':cells})
            else:omitted.append({'kind':'native_table','index':index,'bbox':box,'reason':'native_object_outside_page'})
        for index,item in enumerate(page.images):
            box=[item['x0'],item['top'],item['x1'],item['bottom']]
            if valid_box(box,width,height):
                candidates.append({'kind':'embedded_pdf_image','index':index,'bbox':box,'text':nearby(box)})
            else:omitted.append({'kind':'embedded_pdf_image','index':index,'bbox':box,'reason':'native_object_outside_page'})
        # Unruled parsed tables retain quantized provenance. A readable window is
        # bounded by actual PDF word rectangles, never promoted to verified cells.
        estimate=source.get('bbox_points_estimate')
        if source.get('type')=='table' and valid_box(estimate,width,height):
            inside=[w for w in words if w['top']>=estimate[1]-8 and w['bottom']<=estimate[3]+8]
            if inside:
                box=[min(w['x0'] for w in inside),min(w['top'] for w in inside),
                     max(w['x1'] for w in inside),max(w['bottom'] for w in inside)]
                if valid_box(box,width,height):
                    candidates.append({'kind':'source_region_native_word_union','index':0,'bbox':box,
                        'text':' '.join(w['text'] for w in inside),'cells':None})
        if full_page_recovery:
            plan={'status':'full_page_fallback','reason':'one_bounded_original_selected_page_recovery',
                  'question':question,'association_verified':False,'full_page_recovery':True,
                  'viewport_bbox_top_left_points':[0,0,width,height],'dpi':100,
                  'native_paint_filter':paint_audit,'reference_pages_used':False}
        else:
            plan=choose_window(question,source,candidates,width,height)
        plan['omitted_native_objects']=omitted
        plan['quantized_source_region_used_for_word_selection']=source.get('type')=='table' and valid_box(estimate,width,height)
        viewport=plan.get('viewport_bbox_top_left_points',[0,0,width,height])
        from .native_fields import aligned_header, parameter_rows, comparison_records, apply_comparisons, native_cells, grounded_cell_rows
        if plan.get('kind')=='source_region_native_word_union':
            header=aligned_header(words,viewport)
            if header and header['top']<viewport[1]:
                old=list(viewport); viewport[1]=header['top']
                plan['viewport_bbox_top_left_points']=viewport
                recovery_reason=('numeric source window omitted aligned column labels' if header['header_top']<old[1]
                                 else 'source window omitted table title and attribution context')
                plan['same_page_header_recovery']={'count':1,'reason':recovery_reason,
                    'original_viewport':old,'new_viewport':list(viewport),'header':header,
                    'actual_added_text':' '.join(w['text'] for w in words if viewport[1]<=w['top']<old[1]),
                    'maximum_page_reads':1,'reference_labels_used':False}
                pixels144=math.ceil((viewport[2]-viewport[0])*2)*math.ceil((viewport[3]-viewport[1])*2)
                plan['dpi']=144 if pixels144<=1_350_000 else 100
        visible=[w for w in words if w['x0']>=viewport[0] and w['x1']<=viewport[2]
                 and w['top']>=viewport[1] and w['bottom']<=viewport[3]]
        plan['native_word_records']=[{k:w[k] for k in ('text','x0','x1','top','bottom')} for w in visible]
        operators=comparison_records(page.chars,page.edges,viewport)
        native_cells(tables,operators,viewport)
        plan['native_comparison_records']=operators
        corrected=apply_comparisons(visible,operators)
        for record in operators:
            line=[w for w in corrected if abs(w['top']-record['glyph_box'][1])<3]
            record['source_line']=' '.join(w['text'] for w in sorted(line,key=lambda w:w['x0']))
        if plan.get('kind')=='native_table':
            plan['operator_grounded_cell_rows']=grounded_cell_rows(plan['source_cell_rows'],operators,plan['native_object_index'])
        plan['native_parameter_rows']=parameter_rows(visible)
        plan['unmodified_native_visible_text']=' '.join(w['text'] for w in visible)
        visible=corrected
        plan['native_visible_text']=' '.join(w['text'] for w in visible)
        plan['native_spatial_word_lines']=native_word_lines(visible)
        plan['spatial_word_groups_are_semantic_cells']=False
        box=plan.get('native_object_bbox_top_left_points')
        outside=[w for w in visible if box and not (w['x0']>=box[0] and w['x1']<=box[2]
                 and w['top']>=box[1] and w['bottom']<=box[3])]
        plan['native_context_text']=' '.join(w['text'] for w in outside)
        if full_page_recovery:
            from .confusion_fields import native_matrix
            matrix=native_matrix(plan)
            if matrix is not None:
                plan['native_count_matrix']=matrix
            elif native_unit_prose_recovery and source.get('type')=='text':
                from .native_unit_prose import native_unit_prose_plan
                prose=native_unit_prose_plan(question,plan['native_word_records'],width,height,
                    source_box=source.get('bbox_points_estimate'),source_transcription=source.get('text'))
                if prose is not None:
                    plan['native_unit_prose_recovery']=True
                    plan['native_unit_prose_plan']=prose
    return plan
