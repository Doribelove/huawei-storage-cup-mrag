"""Literal native confusion-matrix fields and labelled Decimal derivations.

No benchmark identifiers, reference answers or metric substitutes. Ambiguous
geometry returns no plan; arithmetic never authorizes a semantic support gate.
"""
from decimal import Decimal,localcontext
import re

from .native_fields import word_rows,box

INTEGER=re.compile(r'^\d{1,12}$')
CLASS=re.compile(r'^[A-Za-z][A-Za-z0-9_-]{0,30}$')
AXES={'prediction':'prediction','predicted':'prediction','reference':'reference','actual':'reference'}

def center(word):return (word['x0']+word['x1'])/2

def native_matrix(viewport):
    words=viewport.get('native_word_records',[])
    if not words or len(words)>15000:return None
    if viewport.get('native_paint_filter',{}).get('schema')!='native-form-bbox-paint-filter-v1':return None
    rows=word_rows(words);numeric=[]
    for y,row in rows:
        nums=[w for w in row if INTEGER.fullmatch(w['text'])]
        if not 2<=len(nums)<=8:continue
        left=[w for w in row if w['x1']<nums[0]['x0'] and w['text'].casefold() not in AXES]
        if len(left)!=1 or not CLASS.fullmatch(left[0]['text']):continue
        if any(w['x0']>=nums[0]['x0'] and w not in nums for w in row):continue
        numeric.append((y,left[0],nums))
    possibilities=[]
    for start,first in enumerate(numeric):
        n=len(first[2]);block=numeric[start:start+n]
        if len(block)!=n or any(len(row[2])!=n for row in block):continue
        if any(not 4<b[0]-a[0]<45 for a,b in zip(block,block[1:])):continue
        labels=[row[1]['text'] for row in block]
        if len(set(labels))!=n:continue
        centers=[center(w) for w in first[2]]
        if any(b-a<15 for a,b in zip(centers,centers[1:])):continue
        if any(abs(center(w)-c)>10 for _,_,nums in block for w,c in zip(nums,centers)):continue
        headers=[(hy,row) for hy,row in rows if first[0]-65<hy<first[0]-3 and len(row)==n
            and {w['text'] for w in row}==set(labels)
            and all(abs(center(w)-c)<=10 for w,c in zip(row,centers))]
        if len(headers)!=1:continue
        hy,header=headers[0];columns=[w['text'] for w in header]
        titles=[(ty,row) for ty,row in rows if hy-100<ty<hy and
            re.match(r'table(?:\s|\d|[.:])',' '.join(w['text'] for w in row),re.I) and
            'confusionmatrix' in re.sub(r'[^a-z]','',' '.join(w['text'] for w in row).casefold())]
        if len(titles)!=1:continue
        caption_words=titles[0][1]
        caption=' '.join(w['text'] for w in sorted(caption_words,key=lambda w:w['x0']))
        if re.search(r'%|percent|normaliz',caption,re.I):continue
        row_axis=[w for w in words if w['text'].casefold() in AXES and w['x1']<min(r[1]['x0'] for r in block)
            and first[0]-4<=w['top']<=block[-1][0]+4]
        col_axis=[w for w in words if w['text'].casefold() in AXES and hy-35<=w['top']<hy-3
            and centers[0]-15<=center(w)<=centers[-1]+15]
        if len(row_axis)!=1 or len(col_axis)!=1:continue
        ra,ca=row_axis[0],col_axis[0]
        if AXES[ra['text'].casefold()]==AXES[ca['text'].casefold()]:continue
        values=[[w['text'] for w in row[2]] for row in block]
        if any(sum(map(int,row))==0 for row in values):continue
        if any(sum(int(row[j]) for row in values)==0 for j in range(n)):continue
        possibilities.append({'title':caption,'row_axis_literal':ra['text'],'column_axis_literal':ca['text'],
            'row_axis':AXES[ra['text'].casefold()],'column_axis':AXES[ca['text'].casefold()],
            'row_labels':labels,'column_labels':columns,'literal_counts':values,
            'source_coordinates':{'title':[box(w) for w in caption_words],'row_axis':box(ra),'column_axis':box(ca),
                'row_labels':[box(r[1]) for r in block],'column_labels':[box(w) for w in header],
                'cells':[[box(w) for w in r[2]] for r in block]},
            'field_binding':'one native label and count at each aligned row/column intersection; diagonal matched by class label',
            'visibility_scope':'native Form-bound records plus verified original image; non-Form painting constraints remain unverified'})
    return possibilities[0] if len(possibilities)==1 else None

def _rate(label,total,diagonal,axis):
    wrong=total-diagonal
    with localcontext() as ctx:
        ctx.prec=40
        percentage=Decimal(wrong)*100/Decimal(total)
        rounded=percentage.quantize(Decimal('.01'))
    return {'class_label':label,'conditioning_axis':axis,'misclassified_count':str(wrong),
        'total_count':str(total),'correct_count':str(diagonal),'fraction':str(wrong)+'/'+str(total),
        'percent_exact_decimal':str(percentage),'percent_rounded_2dp':str(rounded),'unit':'percent',
        'operation':'100 * (axis total - matched-label diagonal) / axis total'}

def confusion_rate_plan(question,catalog):
    if not re.search(r'\bmisclassifi(?:cation|ed)\b|\bconfusion\b.{0,50}\brates?\b',question,re.I):return None
    found=[]
    for image_id,source in catalog.items():
        if source.get('role')!='primary_retrieved_region':continue
        matrix=native_matrix(source.get('readable_viewport',{}))
        if matrix is None:continue
        values=[[int(x) for x in row] for row in matrix['literal_counts']]
        row_rates=[];column_rates=[]
        for i,label in enumerate(matrix['row_labels']):
            j=matrix['column_labels'].index(label)
            row_rates.append(_rate(label,sum(values[i]),values[i][j],matrix['row_axis']))
            column_rates.append(_rate(label,sum(row[j] for row in values),values[i][j],matrix['column_axis']))
        transitions=[]
        for i,row_label in enumerate(matrix['row_labels']):
            for j,column_label in enumerate(matrix['column_labels']):
                if row_label==column_label:continue
                reference,prediction=((column_label,row_label) if matrix['row_axis']=='prediction'
                    else (row_label,column_label))
                transitions.append({'reference_class':reference,'predicted_class':prediction,
                    'literal_matrix_value':matrix['literal_counts'][i][j],
                    'source_cell_box':matrix['source_coordinates']['cells'][i][j]})
        found.append({'schema':'native-labelled-confusion-rates-v1','image_id':image_id,
            'doc_sha256':source['doc_sha256'],'page':source['page'],'source_image_sha256':source['image']['sha256'],
            **matrix,'row_conditioned_rates':row_rates,'column_conditioned_rates':column_rates,
            'source_observed_off_diagonal_cells':transitions,
            'off_diagonal_scope':'Literal source entries with explicit reference and prediction labels; no inferred destination or generated quotation.',
            'denominator_scope':'Both conditional views are explicit. If question does not choose a denominator, label the view; these are computed rates, not quoted rates.',
            'establishes_support':False,'reference_labels_used':False})
    return found[0] if len(found)==1 else None

def compact_confusion(plan):
    value={k:v for k,v in plan.items() if k not in ('source_coordinates','visibility_scope','field_binding','source_observed_off_diagonal_cells')}
    value['source_observed_off_diagonal_cells']={
        'columns':['reference_class','predicted_class','literal_matrix_value'],
        'rows':[[r[k] for k in ('reference_class','predicted_class','literal_matrix_value')]
                for r in plan['source_observed_off_diagonal_cells']]}
    return value

def _mentions(text,labels):
    found=[]
    for label in labels:
        matches=list(re.finditer(r'(?<![\w-])'+re.escape(label)+r'(?![\w-])',text,0 if len(label)==1 else re.I))
        if len(matches)==1:found.append((matches[0].start(),label))
    return sorted(found)

def _check_percent(plan,label,match,clause,axes):
    expected=[]
    for group in (plan['row_conditioned_rates'],plan['column_conditioned_rates']):
        expected.extend(Decimal(r['percent_exact_decimal']) for r in group
            if r['class_label']==label and (not axes or r['conditioning_axis'] in axes))
    literal=match[1];places=len(literal.split('.')[1]) if '.' in literal else 0
    tolerance=Decimal('0.5')*(Decimal(10)**-places)
    return {'kind':'native_labelled_misclassification_percentage','class_label':label,
        'passed':any(abs(Decimal(literal)-v)<=tolerance for v in expected),
        'candidate_literal_percent':literal,'candidate_clause':clause,
        'candidate_conditioning_axes':sorted(axes),'allowed_conditioned_percentages':[str(v) for v in expected],
        'rounding_tolerance_percent':str(tolerance),'establishes_support':False,
        'reason':'percentage must follow actual matched-label off-diagonal count and explicitly scoped denominator'}

def rate_checks(plan,answer):
    """Check only explicit class-associated percentage claims; no positive proof."""
    checks=[];labels=plan['row_labels']
    clauses=re.split(r';|\n|(?<=[.!?])\s+(?=[A-Z])',answer)
    for clause in clauses:
        explicit=bool(re.search(r'\bmisclassifi\w*|\berror\s+rates?\b',clause,re.I))
        if not explicit and (not re.search(r'\brates?\b',clause,re.I)
                or re.search(r'\b(?:accuracy|precision|recall|F1)\b',clause,re.I)):continue
        percents=list(re.finditer(r'(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*%',clause))
        scopes=list(re.finditer(r'\b(?:when\s+)?conditioned\s+on\s+(prediction|predicted|reference|actual)\b',clause,re.I))
        if scopes and percents:
            prefix=clause[:percents[0].start()];shared=_mentions(prefix,labels)
            spans=[];previous=0;seen=set()
            if scopes[0].start()>percents[0].end():
                for scope in scopes[:4]:
                    spans.append((previous,scope.start(),scope[1]));previous=scope.end()
            else:
                for i,scope in enumerate(scopes[:4]):
                    end=scopes[i+1].start() if i+1<len(scopes) else len(clause)
                    spans.append((scope.end(),end,scope[1]))
            for start,end,axis_literal in spans:
                group=[m for m in percents if start<=m.start()<end]
                text=clause[start:end];local=_mentions(text,labels)
                mentions=local or shared
                alternating=bool(local) and all(pos<m.start()-start and
                    (i==len(local)-1 or m.end()-start<local[i+1][0])
                    for i,((pos,_),m) in enumerate(zip(local,group)))
                if (mentions and len(mentions)==len(group) and
                        (len(mentions)==1 or 'respectively' in text.casefold()
                         or alternating)):
                    axis={AXES[axis_literal.casefold()]}
                    for (_,label),m in zip(mentions,group):
                        checks.append(_check_percent(plan,label,m,clause,axis));seen.add(m.start())
            if seen:continue
        mentioned=_mentions(clause,labels)
        if not mentioned or len(mentioned)!=len(percents):continue
        # Multi-class claims need either alternating label/value order or an
        # explicit respectively list. Other prose relations remain unverified.
        alternating=all(pos<m.start() and (i==len(mentioned)-1 or m.end()<mentioned[i+1][0])
            for i,((pos,label),m) in enumerate(zip(mentioned,percents)))
        if len(mentioned)>1 and not alternating and 'respectively' not in clause.casefold():continue
        axes={AXES[m.group().casefold()] for m in re.finditer(r'\b(?:prediction|predicted|reference|actual)\b',clause,re.I)}
        for (_,label),m in zip(mentioned,percents):
            checks.append(_check_percent(plan,label,m,clause,axes if len(axes)==1 else set()))
    return checks
