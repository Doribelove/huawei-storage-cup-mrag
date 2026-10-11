"""Source geometry only: preserve labels, cells and drawn comparison boundaries.

Coordinates are native PDF points (top-left). No question IDs, answer labels or
reference pages. Spatial associations remain distinct from semantic judgments.
"""
import re

SCALAR = re.compile(r'^[+−-]?\d+(?:[,.]\d+)*%?$')


def word_rows(words):
    rows = []
    for word in sorted(words, key=lambda w: (w['top'], w['x0'])):
        if not rows or abs(word['top'] - rows[-1][0]) > 3:
            rows.append([word['top'], [word]])
        else:
            rows[-1][1].append(word)
    return [(y, sorted(row, key=lambda w:w['x0'])) for y,row in rows]


def box(word):
    return [round(word[k], 3) for k in ('x0','top','x1','bottom')]


def aligned_header(words, viewport):
    """Locate a same-page header aligned to a visible scalar row, or abstain."""
    rows = word_rows(words)
    numeric = [(y, [w for w in row if SCALAR.fullmatch(w['text'])])
               for y,row in rows if viewport[1] <= y <= viewport[3]]
    numeric = [(y,row) for y,row in numeric if len(row) >= 3]
    if not numeric:
        return None
    y, values = numeric[0]
    centers = [(w['x0']+w['x1'])/2 for w in values]
    possibilities = []
    for hy,row in rows:
        if not max(0, y-500) <= hy < y-3:
            continue
        # One explicit, nonnumeric label per visible numeric column. Captions
        # and prose do not qualify merely by containing enough words.
        labels = [w for w in row if w['x0'] >= values[0]['x0']-20]
        if len(labels) != len(values) or any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*',w['text']) for w in labels):
            continue
        if any(abs((w['x0']+w['x1'])/2-c) > 16 for w,c in zip(labels,centers)):
            continue
        possibilities.append((hy,labels))
    if not possibilities:
        return None
    hy, labels = possibilities[-1]
    titles = [(ty,row) for ty,row in rows if hy-180 <= ty < hy
              and row and re.fullmatch(r'Table', row[0]['text'], re.I)]
    title = titles[-1] if titles else None
    # An intervening separately titled table invalidates any earlier header.
    if any(hy < ty < y and row and row[0]['text'].casefold() == 'table' for ty,row in rows):
        return None
    return {'header_labels':[w['text'] for w in labels],
            'header_boxes':[box(w) for w in labels], 'header_top':hy,
            'title':' '.join(w['text'] for w in title[1]) if title else None,
            'top':max(0,(title[0] if title else hy)-18),
            'association':'same-page numeric-column center alignment; no cross-page inheritance'}


def parameter_rows(words):
    """Keep scalar estimates beside their actual Greek row labels and headers.

    Subscripts may have a different baseline; join only adjacent label glyphs.
    Confidence intervals stay in the ordinary native transcription.
    """
    result=[]
    for y,row in word_rows(words):
        values=[w for w in row if SCALAR.fullmatch(w['text'])]
        if len(values)<3:
            continue
        labels=[w for w in words if w['x1']<values[0]['x0'] and
                y-3 <= w['top'] <= y+15 and re.fullmatch(r'[α-ωΑ-Ω]|\d{1,2}',w['text'])]
        greek=[w for w in labels if re.fullmatch(r'[α-ωΑ-Ω]',w['text'])]
        if len(greek)!=1:
            continue
        root=greek[0]
        subs=[w for w in labels if w['text'].isdigit() and
              -1 <= w['x0']-root['x1'] <= 4 and abs(w['top']-root['top'])<=7]
        if len(subs)>1:
            continue
        label=root['text']+(''.join(w['text'] for w in subs))
        header=aligned_header(words,[0,y-1,max(w['x1'] for w in words),y+1])
        if header is None or len(header['header_labels'])!=len(values):
            continue
        result.append({'row_label':label,'row_label_boxes':[box(root)]+[box(w) for w in subs],
                       'columns':header['header_labels'],'header_boxes':header['header_boxes'],
                       'values':[w['text'] for w in values],'value_boxes':[box(w) for w in values],
                       'association':'native horizontal estimate row and adjacent label glyphs'})
    return result


def comparison_records(chars, edges, viewport):
    result=[]
    for c in chars:
        if c['text'] not in ('>','<','≥','≤') or not (viewport[0]<=c['x0'] and c['x1']<=viewport[2]
                and viewport[1]<=c['top'] and c['bottom']<=viewport[3]):
            continue
        width=c['x1']-c['x0']
        underlines=[e for e in edges if abs(e['top']-e['bottom'])<.1
                    and .8*width <= e['x1']-e['x0'] <= 1.8*width
                    and abs(e['x0']-c['x0'])<=1 and e['x1']>=c['x1']-.5
                    and c['bottom']-2.5<=e['top']<=c['bottom']+1.5]
        operator=c['text']
        if underlines and operator in ('>','<'):
            operator={'>' :'≥','<':'≤'}[operator]
        result.append({'extracted_glyph':c['text'],'operator':operator,'glyph_box':box(c),
                       'underline_boxes':[box(e) for e in underlines],
                       'method':'literal glyph plus short native underline' if underlines else 'literal native glyph'})
    return result


def apply_comparisons(words, records):
    """A separate transcription; original word records are retained unchanged."""
    output=[dict(w) for w in words]
    for record in records:
        if record['operator']==record['extracted_glyph']:
            continue
        x,y,right,bottom=record['glyph_box']
        matches=[w for w in output if abs(w['x0']-x)<.1 and abs(w['top']-y)<.1
                 and w['text']==record['extracted_glyph']]
        if len(matches)==1:
            matches[0]['text']=record['operator']
    return output


def native_cells(tables, records, viewport):
    for table_index,table in enumerate(tables):
        extracted=table.extract()
        for ri,row in enumerate(table.rows):
            for ci,cell in enumerate(row.cells):
                if cell is None:
                    continue
                x,y,right,bottom=cell
                if not (viewport[0]<=x and right<=viewport[2] and viewport[1]<=y and bottom<=viewport[3]):
                    continue
                for record in records:
                    gx,gy,gr,gb=record['glyph_box']
                    if x<=gx and gr<=right and y<=gy and gb<=bottom:
                        record.update(table_index=table_index,row_index=ri,column_index=ci,
                                      cell_box=list(cell),row_label=extracted[ri][0])


def grounded_cell_rows(cells,records,table_index):
    """Restore only a uniquely matched drawn boundary within its actual cell."""
    if cells is None:return None
    result=[list(row) for row in cells]
    for record in records:
        if record.get('table_index')!=table_index or record['operator']==record['extracted_glyph']:
            continue
        ri=record['row_index'];ci=record['column_index'];text=result[ri][ci]
        if not isinstance(text,str):continue
        value=re.search(re.escape(record['operator'])+r'\s*([+-]?\d+(?:\.\d+)?)',record.get('source_line',''))
        if value is None:continue
        pattern=re.escape(record['extracted_glyph'])+r'(\s*)'+re.escape(value[1])+r'(?![.\d])'
        if len(re.findall(pattern,text))==1:
            result[ri][ci]=re.sub(pattern,lambda m:record['operator']+m[1]+value[1],text)
    return result
