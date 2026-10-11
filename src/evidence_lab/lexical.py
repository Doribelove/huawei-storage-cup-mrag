from __future__ import annotations

from collections import Counter
import math
import re
from .safe_data import evidence_output

STOP={'what','which','is','are','the','a','an','of','to','in','on','for','and','does','how','with','from','by','was','were'}


def tokens(text: str) -> list[str]:
    pieces=re.findall(r'[a-z0-9]+|[\u3400-\u9fff]+',text.casefold())
    result=[]
    for piece in pieces:
        if re.fullmatch(r'[\u3400-\u9fff]+',piece):
            result.extend(piece[i:i+2] for i in range(max(1,len(piece)-1)))
        elif piece not in STOP: result.append(piece)
    return result


class BM25:
    """Diagnostic lexical route over trusted-role evidence; no embedding substitute."""
    def __init__(self,records:list[dict],k1:float=1.5,b:float=0.75):
        self.records=[evidence_output(record) for record in records]
        self.k1,self.b=k1,b
        self.counts=[Counter(tokens(r['text'])) for r in self.records]
        self.lengths=[sum(c.values()) for c in self.counts]
        self.average=sum(self.lengths)/max(len(records),1)
        self.df=Counter(term for c in self.counts for term in c)

    def search(self,query:str,top_k:int=5)->list[dict]:
        if not 1<=top_k<=100: raise ValueError('top_k must be 1..100')
        qt=set(tokens(query))
        scored=[]
        for index,counts in enumerate(self.counts):
            score=0.0
            for term in qt:
                tf=counts.get(term,0)
                if not tf: continue
                idf=math.log(1+(len(self.records)-self.df[term]+0.5)/(self.df[term]+0.5))
                denominator=tf+self.k1*(1-self.b+self.b*self.lengths[index]/(self.average if self.average else 1))
                score+=idf*tf*(self.k1+1)/denominator
            if score>0: scored.append((score,index))
        scored.sort(key=lambda x:(-x[0],self.records[x[1]]['evidence_id']))
        return [{'route':'bm25','score':round(score,6),**self.records[index]} for score,index in scored[:top_k]]
