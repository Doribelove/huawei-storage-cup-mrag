"""Small source-derived field/period audit; no question IDs or evaluation labels."""
from decimal import Decimal, InvalidOperation
import re

METRIC_ALIASES = {
    'retained earnings': ('retained earnings', 'accumulated profits', 'accumulated earnings'),
    'share capital': ('share capital',),
    'share premium': ('share premium',),
    'non controlling interests': ('non controlling interests', 'non-controlling interests'),
}
MONTHS = {name.lower(): n for n, name in enumerate(
    ['January','February','March','April','May','June','July','August','September','October','November','December'], 1)}
NUMBER = re.compile(r'^\(?[+-]?\d[\d,]*(?:\.\d+)?\)?$')

def decimal_value(text):
    text=str(text).strip().replace('−','-')
    if not NUMBER.fullmatch(text): raise ValueError('Not one native numeric field')
    negative=text.startswith('(') and text.endswith(')')
    value=Decimal(text.strip('()').replace(',',''))
    return -value if negative else value

def normalized(text):
    return ' '.join(re.findall(r'\w+',str(text).casefold()))

def _word_rows(words):
    rows=[]
    for word in sorted(words,key=lambda w:(w['top'],w['x0'])):
        if not rows or abs(word['top']-rows[-1][0])>3:rows.append([word['top'],[word]])
        else:rows[-1][1].append(word)
    return [(y,sorted(row,key=lambda w:w['x0'])) for y,row in rows]

def balance_fields(viewport):
    words=viewport.get('native_word_records',[])
    if not words or len(words)>15000:return []
    records=[]
    for y,row in _word_rows(words):
        texts=[w['text'] for w in row];label=' '.join(texts)
        match=re.match(r'^Balance at (\d{1,2}) ([A-Za-z]+) (\d{4})\b',label,re.I)
        if not match or match[2].lower() not in MONTHS:continue
        yearword=next((i for i,w in enumerate(row) if w['text']==match[3]),None)
        nums=row[yearword+1:] if yearword is not None else []
        if len(nums)<2 or any(not NUMBER.fullmatch(w['text']) for w in nums):continue
        records.append({'date':(int(match[3]),MONTHS[match[2].lower()],int(match[1])),
            'row_label':' '.join(texts[:yearword+1]),'y':y,'words':nums,
            'row_label_boxes':[[w[k] for k in ('x0','top','x1','bottom')] for w in row[:yearword+1]]})
    if len(records)<2:return []
    first=min(records,key=lambda x:x['y']);centers=[(w['x0']+w['x1'])/2 for w in first['words']]
    if any(b-a<15 for a,b in zip(centers,centers[1:])):return []
    headers=[[] for _ in centers];header_words=[[] for _ in centers]
    for w in sorted(words,key=lambda x:(x['top'],x['x0'])):
        if not first['y']-75<=w['top']<first['y']-3:continue
        center=(w['x0']+w['x1'])/2
        index=min(range(len(centers)),key=lambda i:abs(center-centers[i]))
        if abs(center-centers[index])<=35:
            headers[index].append(w['text']);header_words[index].append(w)
    result=[]
    for record in records:
        if len(record['words'])!=len(centers):continue
        for i,w in enumerate(record['words']):
            if abs((w['x0']+w['x1'])/2-centers[i])>20:continue
            result.append({'column':' '.join(headers[i]),'row':record['row_label'],
                'date':record['date'],'literal':w['text'],'value':decimal_value(w['text']),
                'column_index':i,'unit':'€' if '€' in headers[i] else None,
                'value_box':[w[k] for k in ('x0','top','x1','bottom')],
                'row_label_boxes':record['row_label_boxes'],
                'column_label_boxes':[[h[k] for k in ('x0','top','x1','bottom')] for h in header_words[i]]})
    return result

def financial_change_plan(question,catalog):
    q=normalized(question)
    metric=next((k for k,aliases in METRIC_ALIASES.items() if any(normalized(a) in q for a in aliases)),None)
    if metric is None or not re.search(r'\b(shift|change|difference|increased?|decreased?|reduction)\b',q):return None
    years=[int(x) for x in re.findall(r'\b(?:19|20)\d{2}\b',question)]
    if len(years)!=2:return None
    possibilities=[]
    for image_id,source in catalog.items():
        fields=[f for f in balance_fields(source.get('readable_viewport',{}))
                if metric in normalized(f['column'])]
        if not fields:continue
        if len({f['column_index'] for f in fields})!=1:continue
        starts=[f for f in fields if f['date'][0]==years[0]]
        ends=[f for f in fields if f['date'][0]==years[1]]
        if re.search(r'\bmid\b',q):starts=[f for f in starts if f['date'][1:] in [(7,1),(6,30)]]
        if re.search(r'\bend\b',q):ends=[f for f in ends if f['date'][1:]==(12,31)]
        if len(starts)!=1 or len(ends)!=1:continue
        a=starts[0];b=ends[0]
        if a['date']>=b['date'] or a['unit']!=b['unit']:continue
        delta=b['value']-a['value']
        possibilities.append({'schema':'native-matched-column-change-v1','image_id':image_id,
           'metric':metric,'literal_column':a['column'],'column_index':a['column_index'],
           'start_row':a['row'],'start_value':str(a['value']),'end_row':b['row'],'end_value':str(b['value']),
           'operation':'end - start','signed_change':str(delta),'unit':a['unit'],
           'operand_coordinates':{'start':a['value_box'],'end':b['value_box'],
                'column_labels':a['column_label_boxes'],'start_row':a['row_label_boxes'],'end_row':b['row_label_boxes']},
           'source_image_sha256':source['image']['sha256'],
           'scope':'actual visible native word coordinates; field alias explicit; arithmetic exact Decimal'})
    return possibilities[0] if len(possibilities)==1 else None

def compact_derivation(plan):
    """Full coordinate topology stays in audit; prompt keeps only operand boxes."""
    if plan.get('schema')=='native-labelled-confusion-rates-v1':
        from .confusion_fields import compact_confusion
        return compact_confusion(plan)
    value={k:v for k,v in plan.items() if k!='operand_coordinates'}
    value['operand_boxes']={k:v for k,v in plan.get('operand_coordinates',{}).items() if k in ('start','end')}
    return value


def caption_context_conflicts(verdict,question):
    """A scene caption alone cannot certify a requested visual object identity.

    Audit reason scope only; never invent an object inventory or answer phrase.
    Explicit depicted-object reasons remain for semantic source verification.
    """
    if not re.search(r'\b(machinery|machines|equipment)\b',question,re.I):return []
    conflicts=[]
    for item in verdict['claims']:
        reason=item['reason']
        caption_context=bool(re.search(r'caption.{0,200}(?:matches?.{0,50}context|context.{0,50}matches?)',reason,re.I))
        actual_identity=bool(re.search(r'\b(?:depicts?|visible|shows? (?:a|an|the))\b',reason,re.I))
        if item['supported'] and caption_context and not actual_identity:
            conflicts.append({'claim_id':item['id'],'type':'caption_context_without_requested_visual_identity'})
    return conflicts


def deterministic_audit(question,answer,catalog):
    """Negative checks only. Passing never establishes semantic support."""
    checks=[]
    # A disaster motivating reconstruction does not prove material performance.
    # This negative audit is restricted to text regions with actual native text;
    # it does not judge image-only or diagram performance claims by text absence.
    performance_claims=re.findall(r'\b(earthquake|seismic|fire|water)[ -](safe|resistant|proof)\b',answer,re.I)
    text_sources=[(key,s) for key,s in catalog.items() if s.get('source_region_type')=='text'
                  and len(s.get('readable_viewport',{}).get('native_visible_text',''))>=100]
    if performance_claims and len(text_sources)==len(catalog) and text_sources:
        for hazard,property_name in performance_claims:
            hazard_pattern=r'(?:earthquake|seismic)' if hazard.lower() in ('earthquake','seismic') else re.escape(hazard)
            proof_pattern=(hazard_pattern+r'[ -](?:safe|resistan\w*|proof|safety|performance|design)\b|'
                           r'\b(?:resistan\w*|safety|performance)\b[^.!?\n]{0,35}'+hazard_pattern)
            matches=[]
            for key,s in text_sources:
                native=s['readable_viewport']['native_visible_text']
                matches.extend({'image_id':key,'source_phrase':m.group(0)} for m in re.finditer(proof_pattern,native,re.I))
            checks.append({'kind':'explicit_material_performance_qualification','passed':bool(matches),
                           'candidate_qualification':hazard+'-'+property_name,'native_source_phrases':matches,
                           'reason':'a named material safety/resistance property needs explicit source performance evidence; disaster context is insufficient',
                           'establishes_support':False})
    plan=financial_change_plan(question,catalog)
    if plan:
        numbers=[]
        for x in re.findall(r'(?<!\w)\(?[+-]?\d[\d,]*(?:\.\d+)?\)?',answer.replace('−','-')):
            try:numbers.append(decimal_value(x))
            except (ValueError,InvalidOperation):pass
        expected=Decimal(plan['signed_change'])
        signed_ok=expected in numbers
        decrease=bool(re.search(r'\b(decreas\w*|fell|fall|drop\w*|reduc\w*)\b',answer,re.I))
        increase=bool(re.search(r'\b(increas\w*|rose|rise|grow\w*)\b',answer,re.I))
        directional_ok=(abs(expected) in numbers and ((expected<0 and decrease and not increase)
                          or (expected>0 and increase and not decrease)))
        allowed={expected,abs(expected),Decimal(plan['start_value']),Decimal(plan['end_value'])}
        allowed.update(Decimal(x) for x in re.findall(r'\b(?:19|20)\d{2}\b',question))
        unit_ok=not (plan['unit']=='€' and re.search(r'\$|\b(?:USD|GBP|dollars?|pounds?)\b',answer,re.I))
        accepted=(signed_ok or directional_ok) and all(n in allowed for n in numbers) and unit_ok
        checks.append({'kind':'matched_metric_period_decimal_change','passed':accepted,'plan':plan,
                       'reason':'exact source operands, signed change and unit required'})
    rows=[r for s in catalog.values() for r in s.get('readable_viewport',{}).get('native_parameter_rows',[])]
    for clause in re.split(r';|\n|(?<=[.])\s+(?=[A-Z])',answer):
        clause=clause.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹','01234567890123456789'))
        labels=re.findall(r'([α-ωΑ-Ω])\s*[_₂₁₃]?\s*(\d+)',clause)
        if not labels:continue
        label=''.join(labels[0]);matches=[r for r in rows if r['row_label']==label]
        tail=re.sub(r'[α-ωΑ-Ω]\s*[_₂₁₃]?\s*\d+','',clause)
        values=re.findall(r'(?<!\w)[+-]?\d+\.\d+(?!\w)',tail)
        if not matches or not values:continue
        possible={decimal_value(x) for r in matches for x in r['values']}
        checks.append({'kind':'native_parameter_row_values','row_label':label,
                       'passed':all(decimal_value(v) in possible for v in values),
                       'source_rows':matches,'reason':'numbers must belong to the claimed row under actual column labels'})
    qwords=set(re.findall(r'\w+',question.casefold()))
    boundary_pattern=r'(>=|<=|≥|≤|>|<|at least|at most|more than|less than)\s*(\d+(?:\.\d+)?)\s*(weeks?|months?|days?)'
    aliases={'at least':'≥','at most':'≤','more than':'>','less than':'<','>=':'≥','<=':'≤'}
    clauses=[c.strip() for c in re.split(r';|\n|(?<=[.])\s+(?=[A-Z])',answer) if c.strip()]
    records=[(image_id,r) for image_id,s in catalog.items()
             for r in s.get('readable_viewport',{}).get('native_comparison_records',[])]
    row_labels={frozenset(re.findall(r'\w+',str(r.get('row_label','')).casefold())) for _,r in records}
    row_labels.discard(frozenset())
    scoped=[(c,{label for label in row_labels if label<=set(re.findall(r'\w+',c.casefold()))}) for c in clauses]
    explicit_rows=any(labels for _,labels in scoped)
    for image_id,s in catalog.items():
        for r in s.get('readable_viewport',{}).get('native_comparison_records',[]):
            row_words=set(re.findall(r'\w+',str(r.get('row_label','')).casefold()))
            if not row_words:continue
            # An operator belonging to a named row must not be compared with
            # another answer sentence's threshold. Mixed-row clauses remain
            # for semantic verification; layout/lexical ambiguity is no proof.
            relevant=[c for c,labels in scoped if labels=={frozenset(row_words)}]
            scope='literal_candidate_clause_named_native_row'
            if not explicit_rows and row_words<=qwords:
                relevant=clauses;scope='unlabeled_answer_question_row'
            if not relevant:continue
            declared=[(aliases.get(op.casefold(),op),decimal_value(n),u.casefold().rstrip('s'),clause)
                      for clause in relevant for op,n,u in re.findall(boundary_pattern,clause,re.I)]
            for op,n,u in re.findall(boundary_pattern,r.get('source_line',''),re.I):
                expected_op=aliases.get(op.casefold(),op);n=decimal_value(n);u=u.casefold().rstrip('s')
                comparable=[v for v in declared if v[1:3]==(n,u)]
                if comparable:
                    checks.append({'kind':'native_comparison_boundary','passed':all(v[0]==expected_op for v in comparable),
                                   'image_id':image_id,'source_record':r,
                                   'candidate_clauses':list(dict.fromkeys(v[3] for v in comparable)),
                                   'row_binding_scope':scope,
                                   'reason':'strict/inclusive boundary and unit must match actual native glyph/underline'})
    if plan is None:
        from .confusion_fields import confusion_rate_plan,rate_checks
        plan=confusion_rate_plan(question,catalog)
        if plan:checks.extend(rate_checks(plan,answer))
    return {'applicable':bool(checks),'passed':all(c['passed'] for c in checks),'checks':checks,
            'plan':plan,'reason':'; '.join(c['reason'] for c in checks if not c['passed']) or 'No deterministic contradiction found',
            'establishes_support':False,'reference_labels_used':False}
