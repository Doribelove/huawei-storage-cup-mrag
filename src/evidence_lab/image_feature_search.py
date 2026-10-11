"""Sparse search hints that return original source records, never hint quotes."""
from collections import Counter
import copy
import hashlib
import json
import math

from .lexical import tokens


def source_record_hash(source):
    return hashlib.sha256(json.dumps(source,ensure_ascii=False,sort_keys=True,
        separators=(',',':')).encode()).hexdigest()


class ImageFeatureSearch:
    def __init__(self,features,canonical_sources):
        self.canonical={r['evidence_id']:r for r in canonical_sources}
        self.rows=[];self.failed_feature_records=0
        seen=set()
        for feature in features:
            eid=feature['evidence_id']
            if eid in seen:
                raise ValueError('Duplicate image feature binding')
            seen.add(eid)
            source=self.canonical.get(eid)
            if source is None or source['type']!='image':
                raise ValueError('Search feature must map to an original image region')
            if (source['document_name'],source['page'])!=(feature['document_name'],feature['page']):
                raise ValueError('Search feature document/page differs from original')
            if feature['status']=='failed':
                self.failed_feature_records+=1
                continue
            if (feature.get('source_quote_or_answer_fact_established') is not False
                    or feature.get('canonical_source_text_replaced') is not False
                    or feature.get('doc_sha256')!=source['doc_sha256']
                    or feature.get('original_source_record_sha256')!=source_record_hash(source)):
                raise ValueError('Unverified feature role or original record binding')
            text=feature.get('search_feature_text')
            if not isinstance(text,str) or len(text)>12000:
                raise ValueError('Search feature text budget exceeded')
            self.rows.append({'feature':feature,'counts':Counter(tokens(text))})
        self.df=Counter(term for row in self.rows for term in row['counts'])
        self.average=sum(sum(row['counts'].values()) for row in self.rows)/max(1,len(self.rows))

    def search(self,question,top_k=5):
        if not isinstance(question,str) or not question.strip() or len(question)>1000:
            raise ValueError('Question text budget exceeded')
        if type(top_k) is not int or not 1<=top_k<=20:
            raise ValueError('Bounded image feature search requires1..20 matches')
        query=set(tokens(question));ranked=[]
        for row in self.rows:
            score=0;counts=row['counts'];length=sum(counts.values())
            matched=sorted(query & counts.keys())
            for term in matched:
                tf=counts[term]
                idf=math.log(1+(len(self.rows)-self.df[term]+.5)/(self.df[term]+.5))
                score+=idf*tf*2.5/(tf+1.5*(.25+.75*length/(self.average or 1)))
            if score>0:
                ranked.append((score,row['feature']['evidence_id'],row,matched))
        ranked.sort(key=lambda r:(-r[0],r[1]))
        result=[]
        for score,eid,row,matched in ranked[:top_k]:
            # No feature text is placed in source.text or an exact quote field.
            # Downstream acquisition must explicitly take the canonical record.
            result.append({'route':'native_image_search_hint_v1','feature_score':round(score,6),
                'matched_query_tokens':matched,'canonical_source':copy.deepcopy(self.canonical[eid]),
                'feature_provenance':{'native_page_record_sha256':row['feature']['page_record_sha256'],
                    'original_source_record_sha256':row['feature']['original_source_record_sha256'],
                    'native_region_word_count':row['feature']['native_region_word_count'],
                    'search_cue_is_source_fact':False,'source_quote_established':False},
                'semantic_support_established':False})
        return result

    def supplementary_source(self,question,existing_sources):
        """One rare-cue source proposal; overlap never establishes support."""
        existing={(s['doc_sha256'],s['page']) for s in existing_sources}
        generic={'after','before','over','under','between','during','according','figure',
                 'table','chart','shown','shows','following','based','their','these',
                 'this','that','using','used','can','could','would','should','each',
                 'both','different','respective','report','reported','years','year'}
        query={term for term in tokens(question) if len(term)>1 and not term.isdigit()
               and term not in generic}
        limit=max(3,int(len(self.rows)*.05))
        rare={term for term in query if 0<self.df[term]<=limit}
        by_id={row['feature']['evidence_id']:row['feature'] for row in self.rows}
        for proposal in self.search(question,top_k=20):
            source=proposal['canonical_source']
            if (source['doc_sha256'],source['page']) in existing:
                continue
            feature=by_id[source['evidence_id']]
            words=feature.get('native_words_in_region',[])
            if len(words)!=feature['native_region_word_count']:
                raise ValueError('Native feature word count differs from actual records')
            native=set(tokens(' '.join(word['text'] for word in words)))
            matched=rare & set(proposal['matched_query_tokens'])
            native_matched=matched & native
            if len(matched)>=2 and native_matched:
                return dict(proposal,acquisition_reason={
                    'rare_query_cues':sorted(matched),'native_region_query_cues':sorted(native_matched),
                    'maximum_cue_document_frequency':limit,
                    'cue_document_frequencies':{term:self.df[term] for term in sorted(matched)},
                    'new_canonical_document_page':True,'support_established':False})
        return None
