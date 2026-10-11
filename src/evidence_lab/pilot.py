"""Deterministic source-only pilot sampling and bounded original-context assembly."""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib
import json
from .safe_data import evidence_output,reject_answer_fields

SEED='pilot-v1-20261003'
BINS=('short','medium','long')


def page_bin(pages:int)->str:
    if isinstance(pages,bool) or not isinstance(pages,int) or pages<1: raise ValueError('Invalid source page count')
    return 'short' if pages<=10 else 'medium' if pages<=20 else 'long'


def sampling_key(doc:dict)->str:
    return hashlib.sha256((SEED+'\0'+doc['sha256']).encode()).hexdigest()


def select_pilot(documents:dict,reviews:list[dict])->tuple[list[dict],dict]:
    """Eight domain slots, one source table-gap slot, one page-bin balancing slot."""
    candidates=[]
    for name,doc in sorted(documents.items()):
        reject_answer_fields(doc)
        candidates.append({'name':name,'domain':name.rsplit('_',1)[0],'sha256':doc['sha256'],
            'page_count':doc['page_count'],'page_bin':page_bin(doc['page_count']),
            'parsed_body_sha256':doc['parsed_body_sha256'],'sampling_key':sampling_key(doc)})
    domains=sorted({d['domain'] for d in candidates})
    if len(domains)!=8 or len(candidates)<10: raise ValueError('Pilot v1 expects the frozen eight-domain source dataset')
    for review in reviews: reject_answer_fields(review)
    gaps={r['document_name'] for r in reviews if r['visual_classification']=='parse-missing'
        and any(a.get('source_type',a.get('type'))=='table' or a.get('kind')=='empty_table' for a in r['anomalies'])}
    if not gaps: raise ValueError('No independently reviewed source table-gap stratum')
    selected=[];used_names=set();used_bodies=set()
    def available(items): return [d for d in items if d['name'] not in used_names and d['parsed_body_sha256'] not in used_bodies]
    def pick(items,slot,target):
        items=available(items)
        if not items: raise ValueError('Sampling stratum exhausted')
        chosen=min(items,key=lambda d:(d['sampling_key'],d['name']))
        selected.append({**chosen,'selection_slot':slot,'target_page_bin':target})
        used_names.add(chosen['name']);used_bodies.add(chosen['parsed_body_sha256'])
    for i,domain in enumerate(domains):
        target=BINS[i%3]
        group=available([d for d in candidates if d['domain']==domain])
        priority=(target,)+tuple(b for b in BINS if b!=target)
        chosen_bin=next((b for b in priority if any(d['page_bin']==b for d in group)),None)
        pick([d for d in group if d['page_bin']==chosen_bin],f'domain:{domain}',target)
    pick([d for d in candidates if d['name'] in gaps],'source:reviewed_table_parse_gap','any')
    counts=Counter(d['page_bin'] for d in selected)
    remaining=available(candidates)
    balanced=min((b for b in BINS if any(d['page_bin']==b for d in remaining)),key=lambda b:(counts[b],BINS.index(b)))
    pick([d for d in remaining if d['page_bin']==balanced],'source:page_bin_balance',balanced)
    rule={'seed':SEED,'page_bins':{'short':'1..10','medium':'11..20','long':'21+'},
        'domain_slots':'eight sorted domains; target bins short/medium/long cyclically; fallback in fixed bin order',
        'table_gap_slot':'one remaining document with independently reviewed parse-missing table; hash order',
        'balance_slot':'least represented available page bin after nine selections; ties short/medium/long',
        'ordering':'SHA256(seed + NUL + original PDF SHA256), then document name',
        'deduplication':'no repeated name or parsed-body SHA256','query_or_answer_inputs':False,
        'candidate_metadata_sha256':hashlib.sha256(json.dumps(candidates,sort_keys=True).encode()).hexdigest()}
    return selected,rule


class ContextAssembler:
    def __init__(self,records:list[dict]):
        self.records={r['evidence_id']:evidence_output(r) for r in records}
        if len(self.records)!=len(records) or any(r.get('role')!='source_pdf_region' for r in self.records.values()):
            raise ValueError('Context requires unique original PDF region evidence')
        self.order={};self.groups=defaultdict(list)
        for row in self.records.values():
            aliases=[a for a in row.get('aliases',[]) if a.get('document_name')==row['document_name']]
            if len(aliases)!=1: continue
            alias=aliases[0]
            if 'content_list_member' not in alias or 'record_index' not in alias: continue
            key=(row['doc_sha256'],row['document_name'],row['page'],alias['content_list_member'])
            self.groups[key].append((alias['record_index'],row['evidence_id']))
            self.order[row['evidence_id']]=(key,alias['record_index'])
        for group in self.groups.values(): group.sort()

    def assemble(self,hits:list[dict],max_anchors:int=5,max_added:int=8,max_characters:int=6000,candidate_filter=None)->dict:
        if not 1<=max_anchors<=5 or not 0<=max_added<=8 or not 0<=max_characters<=6000: raise ValueError('Context budget exceeded')
        additions=[];links=[];seen={h['evidence_id'] for h in hits[:max_anchors]};characters=0;stops=[]
        for hit in hits[:max_anchors]:
            reject_answer_fields({k:v for k,v in hit.items() if k!='score'})
            anchor=self.records.get(hit['evidence_id'])
            if anchor is None: raise ValueError('Unknown context anchor')
            # Match source binding, not a route's mutable score or debug metadata.
            if any(hit.get(k)!=anchor.get(k) for k in ('doc_sha256','document_name','page','text')):
                raise ValueError('Context anchor source mismatch')
            if candidate_filter is not None and not candidate_filter(anchor):
                stops.append('anchor_excluded_by_source_policy');continue
            # Structural trigger only; it does not classify semantic answer support.
            if len(anchor['text'].strip())>160 or anchor['type']!='text': continue
            ordered=self.order.get(anchor['evidence_id'])
            if not ordered: stops.append('anchor_has_no_unambiguous_original_order');continue
            key,position=ordered
            neighbors=[eid for order,eid in self.groups[key] if order>position][:2]
            for eid in neighbors:
                row=self.records[eid]
                if eid in seen or not row['text'].strip(): continue
                if candidate_filter is not None and not candidate_filter(row): continue
                if len(additions)>=max_added: stops.append('added_region_budget');break
                if characters+len(row['text'])>max_characters: stops.append('character_budget');break
                seen.add(eid);characters+=len(row['text']);additions.append(row)
                links.append({'anchor_evidence_id':anchor['evidence_id'],'added_evidence_id':eid,
                    'relation':'next_original_region_on_same_pdf_page','source_record_index':self.order[eid][1]})
        return {'anchors':[h['evidence_id'] for h in hits[:max_anchors]],'added_regions':additions,'links':links,
            'added_characters':characters,'budget':{'anchors':max_anchors,'added_regions':max_added,'added_characters':max_characters,
                'passes':1},'stop_reasons':sorted(set(stops)),'answer_generated':False}


def answer_support_state(hits:list[dict],citations:list[dict])->dict:
    """No reader exists. Source-valid candidates never imply an answered question."""
    ids={h['evidence_id'] for h in hits}
    byid={c['evidence_id']:c for c in citations}
    if not hits: state='no_candidates'
    elif any(eid not in byid or not byid[eid].get('all_passed') for eid in ids): state='candidate_provenance_failed'
    else: state='candidates_only'
    return {'candidate_state':state,'supported_answer':False,'answer_generated':False,'support_state':'not_established',
        'reason':'No answer reader or answer-support verification has run','similarity_threshold_calibrated':False,
        'message':'未建立答案支持；当前仅展示来源候选证据。'}
