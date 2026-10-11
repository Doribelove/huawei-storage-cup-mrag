from __future__ import annotations

from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
import zipfile

from pypdf import PdfReader

from .safe_data import document_name,image_member,member_json,normalized_bbox,preflight_zip,read_member,record_text,sha256_file,validate_name


def reconcile_status(statuses:dict,document_names:set[str])->dict:
    byfile=defaultdict(list)
    for status_id,record in statuses.items():
        if not isinstance(record,dict): raise ValueError('Malformed status record')
        name=document_name(record.get('file_path',''))
        byfile[name].append({'id':status_id,'status':record.get('status')})
    duplicates=[]
    for status_id,record in statuses.items():
        if record.get('status')!='failed': continue
        seen={status_id}
        current=record
        chain=[]
        while current.get('metadata',{}).get('original_doc_id'):
            target=current['metadata']['original_doc_id']
            if target in seen: raise ValueError('Cycle in duplicate provenance')
            seen.add(target)
            chain.append(target)
            if target not in statuses:
                current={}
                break
            current=statuses[target]
        duplicates.append({'failed_id':status_id,'file_path':record.get('file_path'),
                           'is_duplicate':record.get('metadata',{}).get('is_duplicate'),
                           'duplicate_kind':record.get('metadata',{}).get('duplicate_kind'),
                           'chain':chain,'resolved_file':current.get('file_path'),
                           'resolved_status':current.get('status')})
    processed={name for name,records in byfile.items() if any(r['status']=='processed' for r in records)}
    aliases={document_name(r['file_path']):document_name(r['resolved_file']) for r in duplicates
             if r['is_duplicate'] and r['resolved_status']=='processed'}
    covered=processed|{name for name,target in aliases.items() if target in processed}
    return {'status_record_count':len(statuses),'unique_file_name_count':len(byfile),
            'status_record_distribution':dict(Counter(v.get('status','unknown') for v in statuses.values())),
            'processed_unique_files':len(processed),'duplicate_records':duplicates,
            'file_aliases':aliases,'missing_without_duplicate_resolution':sorted(document_names-processed),
            'unresolved_after_status_aliases':sorted(document_names-covered),
            'status_files_outside_dataset':sorted(set(byfile)-document_names),
            'note':'Aliases are status claims; PDF SHA/content comparison is required independently.'}


def audit_archives(data_path:Path,demo_path:Path,output:Path)->dict:
    start=time.perf_counter()
    preflight={'data':preflight_zip(data_path),'demo':preflight_zip(demo_path)}
    output.mkdir(parents=True,exist_ok=True)
    report={'observed_at_utc':datetime.now(timezone.utc).isoformat(),
            'source_archives':{'data':{'path':str(data_path.resolve()),'sha256':sha256_file(data_path)},
                               'demo':{'path':str(demo_path.resolve()),'sha256':sha256_file(demo_path)}},
            'no_network_or_model_calls':True,'archive_preflight':preflight}
    pdf_docs={}
    evidence={}
    type_counts=Counter()
    image_refs=Counter()
    original_path_kinds=Counter()
    invalid=[]
    region_count=0
    with zipfile.ZipFile(data_path) as archive:
        names=set(archive.namelist())
        for name in names: validate_name(name)
        original_members=sorted(n for n in names if n.startswith('original_pdf_files_100/') and n.endswith('.pdf'))
        for index,name in enumerate(original_members):
            content=read_member(archive,name,32*1024**2)
            digest=hashlib.sha256(content).hexdigest()
            reader=PdfReader(BytesIO(content),strict=False)
            pages=[{'width':float(p.mediabox.width),'height':float(p.mediabox.height),'rotation':int(p.get('/Rotate',0)),
                    'cropbox':[float(x) for x in p.cropbox]} for p in reader.pages]
            pdf_docs[document_name(name)]={'member':name,'bytes':len(content),'sha256':digest,'page_count':len(pages),'pages':pages}
            if (index+1)%25==0: print(f'PDF hash/page checks: {index+1}/{len(original_members)}',flush=True)
        for name in sorted(n for n in names if n.endswith('_content_list.json')):
            doc_name=document_name(name)
            doc=pdf_docs.get(doc_name)
            if not doc: raise ValueError('Parsed document has no original PDF')
            records=member_json(archive,name)
            if not isinstance(records,list): raise ValueError('Content list is not an array')
            original_size=archive.getinfo(name).file_size
            content_sha=hashlib.sha256(read_member(archive,name,2*1024**2)).hexdigest()
            doc['content_list']={'member':name,'sha256':content_sha,'bytes':original_size,'records':len(records)}
            body_digest=hashlib.sha256()
            for idx,record in enumerate(records):
                if not isinstance(record,dict): raise ValueError('Content-list record is not an object')
                text=record_text(record)
                body_digest.update((str(record.get('type'))+'\n'+text+'\n').encode('utf-8'))
                page_idx=record.get('page_idx')
                try:
                    bbox=normalized_bbox(record)
                    if isinstance(page_idx,bool) or not isinstance(page_idx,int) or not 0<=page_idx<doc['page_count']:
                        raise ValueError('Page index outside original PDF')
                    img=image_member(record,name,names)
                except ValueError as error:
                    invalid.append({'document':doc_name,'record_index':idx,'reason':str(error)})
                    continue
                type_counts[record.get('type','unknown')]+=1
                region_count+=1
                if 'img_path' in record:
                    image_refs['declared']+=1
                    image_refs['resolved' if img else 'missing_or_empty']+=1
                    raw=str(record['img_path'])
                    original_path_kinds['absolute' if raw.startswith('/') or ':' in raw else 'relative']+=1
                p=doc['pages'][page_idx]
                evidence_id=hashlib.sha256(json.dumps([doc['sha256'],page_idx,record['bbox'],record.get('type'),text],ensure_ascii=False).encode('utf-8')).hexdigest()
                alias={'document_name':doc_name,'pdf_member':doc['member'],'content_list_member':name,'record_index':idx,'image_member':img}
                if evidence_id in evidence:
                    evidence[evidence_id]['aliases'].append(alias)
                    continue
                evidence[evidence_id]={'role':'source_pdf_region','evidence_id':evidence_id,'doc_sha256':doc['sha256'],
                    'document_name':doc_name,'page':page_idx+1,'page_idx':page_idx,'type':record.get('type'),
                    'source_bbox':record['bbox'],'bbox_normalized':bbox,
                    'bbox_points_estimate':[bbox[0]*p['width'],bbox[1]*p['height'],bbox[2]*p['width'],bbox[3]*p['height']],
                    'bbox_coordinate_system':'MinerU integer 0..1000, top-left; PDF points estimate is quantized',
                    'page_size_points':[p['width'],p['height']],'page_rotation':p['rotation'],
                    'text':text,'image_member':img,'source_archive_sha256':report['source_archives']['data']['sha256'],
                    'aliases':[alias],'limits':['native/OCR parsed source; no new visual inference performed']}
            doc['parsed_body_sha256']=body_digest.hexdigest()
        with (output/'regions.jsonl').open('w',encoding='utf-8') as out:
            for region in evidence.values(): out.write(json.dumps(region,ensure_ascii=False)+'\n')
    pdf_groups=defaultdict(list)
    parsed_groups=defaultdict(list)
    for name,document in pdf_docs.items():
        pdf_groups[document['sha256']].append(name)
        parsed_groups[document.get('parsed_body_sha256')].append(name)
    report['dataset']={'logical_pdf_files':len(pdf_docs),'unique_pdf_sha256':len(pdf_groups),
        'verified_total_pdf_pages':sum(d['page_count'] for d in pdf_docs.values()),
        'duplicate_pdf_groups':[v for v in pdf_groups.values() if len(v)>1],
        'duplicate_parsed_body_groups':[v for v in parsed_groups.values() if len(v)>1],
        'source_region_count':region_count,'deduplicated_evidence_regions':len(evidence),'types':dict(type_counts),
        'image_references':dict(image_refs),'original_image_path_kinds':dict(original_path_kinds),
        'invalid_regions':invalid,'documents':pdf_docs}
    with zipfile.ZipFile(demo_path) as archive:
        name=next(n for n in archive.namelist() if n.endswith('/kv_store_doc_status.json'))
        statuses=member_json(archive,name)
        report['status_reconciliation']=reconcile_status(statuses,set(pdf_docs))
        alias_checks=[]
        for source,target in report['status_reconciliation']['file_aliases'].items():
            alias_checks.append({'source':source,'target':target,
                'pdf_sha_equal':pdf_docs[source]['sha256']==pdf_docs[target]['sha256'],
                'parsed_body_sha_equal':pdf_docs[source]['parsed_body_sha256']==pdf_docs[target]['parsed_body_sha256']})
        report['status_reconciliation']['independent_alias_checks']=alias_checks
        name=next(n for n in archive.namelist() if n.endswith('/vdb_chunks.json'))
        payload=read_member(archive,name,64*1024**2)
        index_path=output/'vdb_chunks.snapshot.json'
        if index_path.exists() and index_path.read_bytes()!=payload: raise ValueError('Snapshot differs; refusing overwrite')
        if not index_path.exists(): index_path.write_bytes(payload)
        report['chunk_snapshot']={'path':str(index_path.resolve()),'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'member':name,'member_crc_validated':True}
        # Only schema and category counts; answer content never enters output corpus.
        query_name=next(n for n in archive.namelist() if n.endswith('/queries_30.json'))
        queries=member_json(archive,query_name)
        report['query_quarantine']={'member':query_name,'sha256':hashlib.sha256(read_member(archive,query_name,2*1024**2)).hexdigest(),
           'count':len(queries),'record_keys':sorted(set(k for q in queries for k in q)),
           'domains':dict(Counter(q.get('domain','unknown') for q in queries)),
           'question_types':dict(Counter(q.get('question_type','unknown') for q in queries)),
           'answer_types':dict(Counter(q.get('answer_type','unknown') for q in queries)),
           'gold_content_not_exported_or_indexed':True}
    report['runtime_seconds']=round(time.perf_counter()-start,3)
    report['evidence_corpus_sha256']=sha256_file(output/'regions.jsonl')
    (output/'data-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report
