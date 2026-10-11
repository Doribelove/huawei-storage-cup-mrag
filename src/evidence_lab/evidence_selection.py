"""Bounded label-free page coverage; rankings and original source fields stay intact."""
def select_distinct_pages(hits, count=2):
    if count not in (1, 2): raise ValueError('Reader supports one or two selected regions')
    if not hits: return []
    selected=[hits[0]];pages={(hits[0]['doc_sha256'],hits[0]['page'])}
    for row in hits[1:]:
        if len(selected)>=count:break
        page=(row['doc_sha256'],row['page'])
        if page not in pages:selected.append(row);pages.add(page)
    if len(selected)<count:
        ids={r['evidence_id'] for r in selected}
        for row in hits[1:]:
            if row['evidence_id'] not in ids:
                selected.append(row);ids.add(row['evidence_id'])
                if len(selected)>=count:break
    return selected
