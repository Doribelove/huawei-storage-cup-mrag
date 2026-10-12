"""Bounded original figure context on already retrieved physical pages.

Question terms in adjacent native text establish a search hint, never the
depicted objects, connections, cell values, or an answer sentence.
"""
from collections import Counter
import math
import re

from .readable_region import tokens, valid_box, native_word_lines
from .safe_data import evidence_output


def native_lines(words):
    """Group complete original words by baseline, retaining every rectangle."""
    rows=[]
    for word in sorted(words,key=lambda w:(w['top'],w['x0'])):
        if not rows or abs(word['top']-rows[-1][0])>3:
            rows.append([word['top'],[word]])
        else:
            rows[-1][1].append(word)
    result=[]
    for _,line in rows:
        line=sorted(line,key=lambda w:w['x0'])
        result.append({'text':' '.join(w['text'] for w in line),
            'bbox_top_left_points':[min(w['x0'] for w in line),min(w['top'] for w in line),
                                   max(w['x1'] for w in line),max(w['bottom'] for w in line)],
            'native_word_records':[{k:w[k] for k in ('text','x0','top','x1','bottom')} for w in line]})
    return result


def adjacent_words(words,box):
    return [w for w in words if w['x1']>=box[0]-18 and w['x0']<=box[2]+18
        and w['bottom']>=box[1]-48 and w['top']<=box[3]+48
        and not (w['x0']>=box[0] and w['x1']<=box[2] and w['top']>=box[1] and w['bottom']<=box[3])]


class SamePageFigureSearch:
    def __init__(self,features,rows):
        from .image_feature_search import ImageFeatureSearch
        ImageFeatureSearch(features,rows)
        self.figures=[r for r in rows if r['type']=='image']
        self.features={f['evidence_id']:f for f in features}
        self.df=Counter()
        for row in rows: self.df.update(tokens(row.get('text','')))
        self.rare_limit=max(3,int(.01*len(rows)))

    def propose(self,question,selected,visual):
        import pdfplumber
        if not isinstance(question,str) or not question.strip() or len(question)>500:
            raise ValueError('Original question-only figure association exceeds budget')
        scopes={(s['document_name'],s['page']) for s in selected}
        present={s['evidence_id'] for s in selected}
        if not 1<=len(scopes)<=3: raise ValueError('Selected page budget exceeded')
        query=tokens(question);measured=[]
        for name,number in sorted(scopes):
            candidates=[r for r in self.figures if r['document_name']==name and r['page']==number
                        and r['evidence_id'] not in present]
            if not candidates: continue
            path=visual.browser.verified_pdf(name)
            with pdfplumber.open(path) as pdf:
                page=pdf.pages[number-1]
                if page.rotation or (page.cropbox is not None and page.mediabox!=page.cropbox): continue
                width,height=page.width,page.height
                words=page.extract_words()
                if len(words)>15000: raise ValueError('Original native word budget exceeded')
            for row in candidates:
                feature=self.features[row['evidence_id']]
                if feature['status']!='collected' or feature['native_region_word_count']!=0: continue
                source=evidence_output(row);box=source.get('bbox_points_estimate')
                if not valid_box(box,width,height): raise ValueError('Canonical figure outside original page')
                fraction=(box[2]-box[0])*(box[3]-box[1])/(width*height)
                shared=sorted(query & tokens(' '.join(w['text'] for w in adjacent_words(words,box))))
                rare=[t for t in shared if self.df[t]<=self.rare_limit]
                if len(shared)<2 or not rare or not .01<=fraction<=.80: continue
                proof=visual.verifier.verify(row['evidence_id'])
                if not proof['all_passed']: raise ValueError('Canonical figure source verification failed')
                measured.append({'canonical_source':source,'shared_question_tokens':shared,
                    'shared_rare_question_tokens':rare,'association_score':len(shared)+2*len(rare),
                    'shared_token_df':{t:self.df[t] for t in shared},'actual_source_verification':proof,
                    'already_retrieved_page':True,'native_region_word_count':0,
                    'association_is_semantic_support':False,'reference_pages_used':False})
        measured.sort(key=lambda r:(-r['association_score'],r['canonical_source']['evidence_id']))
        unique=bool(measured) and (len(measured)==1 or measured[0]['association_score']>1.5*measured[1]['association_score'])
        return (measured[0] if unique else None), {'eligible_candidates':measured,
            'unique_strongest':unique,'maximum_new_figure_views':1,'semantic_support_established':False}
