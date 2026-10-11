"""Bounded, selective extraction and static material review. No archive code runs."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile
from xml.etree import ElementTree as ET

from inspect_archives import issues
from evidence_lab.safe_data import preflight_zip

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'materials' / 'reference'
AUDIT = ROOT / 'materials' / 'audit'
MAX_FILE = 32 * 1024**2
MAX_TOTAL = 100 * 1024**2
total = 0
extractions = []


def bounded_read(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
    if issues(info.filename, info) or info.file_size > limit:
        raise ValueError(f'Refusing unsafe or oversized member: {info.filename}')
    with archive.open(info, 'r') as source:
        data = source.read(limit + 1)
    if len(data) != info.file_size or len(data) > limit:
        raise ValueError('Member length differs from declaration/budget')
    return data


def extract(archive: zipfile.ZipFile, info: zipfile.ZipInfo, archive_id: str) -> Path:
    global total
    data = bounded_read(archive, info, MAX_FILE)
    if total + len(data) > MAX_TOTAL:
        raise ValueError('Extraction budget exhausted')
    total += len(data)
    folder = (DEST / archive_id).resolve()
    target = (folder / PurePosixPath(info.filename.replace('\\', '/'))).resolve()
    if not target.is_relative_to(folder):
        raise ValueError('Target escapes isolated directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError('Existing material differs; no overwrite')
    else:
        with target.open('xb') as output:
            output.write(data)
    extractions.append({'archive_id':archive_id,'member':info.filename,'path':str(target),
                        'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'crc_validated':True})
    return target


def review_docx(path: Path) -> None:
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    preflight_zip(path,max_entries=2000,max_uncompressed_bytes=32*1024**2)
    with zipfile.ZipFile(path) as package:
        if len(package.infolist()) > 2000 or sum(i.file_size for i in package.infolist()) > 32 * 1024**2:
            raise ValueError('DOCX nested ZIP budget exceeded')
        for info in package.infolist():
            if issues(info.filename,info):
                raise ValueError('DOCX unsafe nested member')
        xml=bounded_read(package,package.getinfo('word/document.xml'),4*1024**2)
        if b'<!DOCTYPE' in xml or b'<!ENTITY' in xml:
            raise ValueError('DOCTYPE/entities disallowed')
        document=ET.fromstring(xml)
        body=document.find('w:body',ns)
        paragraphs=[]
        for child in body:
            if child.tag.endswith('}p'):
                text=''.join(t.text or '' for t in child.findall('.//w:t',ns))
                if text: paragraphs.append(text)
            elif child.tag.endswith('}tbl'):
                for row in child.findall('w:tr',ns):
                    paragraphs.append(' | '.join(''.join(t.text or '' for t in cell.findall('.//w:t',ns)) for cell in row.findall('w:tc',ns)))
        (AUDIT/'competition-brief.txt').write_text('\n'.join(f'[P{i:03d}] {text}' for i,text in enumerate(paragraphs,1))+'\n',encoding='utf-8')
        media=[]
        for info in package.infolist():
            if info.filename.startswith('word/media/') and Path(info.filename).suffix.lower() in ('.png','.jpg','.jpeg'):
                media.append(str(extract(package,info,'brief-docx-media')))
        metadata={}
        for part in ['docProps/core.xml','docProps/app.xml']:
            if part in package.namelist():
                tree=ET.fromstring(bounded_read(package,package.getinfo(part),100000))
                for child in tree:
                    key=child.tag.split('}')[-1]
                    if key in ('created','modified','Pages','Words','Application','AppVersion'):
                        metadata[key]=child.text
        (AUDIT/'competition-brief.metadata.json').write_text(json.dumps({'source_docx':str(path),'paragraph_count':len(paragraphs),'media':media,'properties':metadata},ensure_ascii=False,indent=2),encoding='utf-8')


def inspect_query_schema(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict:
    data=json.loads(bounded_read(archive,info,1024**2))
    sample=data[0] if isinstance(data,list) and data else data
    result={'member':info.filename,'container':type(data).__name__,'count':len(data),
            'first_record_keys':list(sample) if isinstance(sample,dict) else None,
            'content_not_exposed_or_indexed':True}
    if isinstance(data,dict):
        result['top_level_value_types']={k:type(v).__name__ for k,v in data.items()}
    return result


def main() -> None:
    DEST.mkdir(parents=True,exist_ok=True)
    schema=[]
    demo_report={}
    for archive_id in ['1785407594927732283','1787621779623419422']:
        inventory=json.loads((AUDIT/f'{archive_id}.inventory.json').read_text(encoding='utf-8'))
        if inventory['issue_count'] or inventory['archive_warnings']:
            raise ValueError('Inventory contains unresolved risk flags')
        preflight_zip(Path(inventory['archive']))
        with zipfile.ZipFile(inventory['archive']) as archive:
            if archive_id.startswith('178762'):
                for info in archive.infolist():
                    name=info.filename
                    if info.is_dir() or '/__pycache__/' in name or '/.vscode/' in name or '.egg-info/' in name:
                        continue
                    basename=PurePosixPath(name).name
                    if name.endswith('.docx'):
                        review_docx(extract(archive,info,archive_id))
                    elif name.endswith(('.py','.sh','.md','.toml')) or basename in ('requirements.txt','LICENSE.txt','env.example'):
                        extract(archive,info,archive_id)
                    elif basename in ('queries_30.json','test_query.json'):
                        schema.append(inspect_query_schema(archive,info))
                    elif basename == 'kv_store_doc_status.json':
                        statuses=json.loads(bounded_read(archive,info,1024**2))
                        demo_report['document_status_count']=len(statuses)
                        demo_report['status_distribution']=dict(Counter(v.get('status','unknown') for v in statuses.values()))
                        demo_report['status_record_keys']=list(next(iter(statuses.values())))
                        demo_report['status_file_paths']=[v.get('file_path') for v in statuses.values()][:3]
                    elif basename.startswith('vdb_') and name.endswith('.json'):
                        with archive.open(info) as source:
                            prefix=source.read(4096).decode('utf-8',errors='replace')
                        match=re.search(r'"embedding_dim"\s*:\s*(\d+)',prefix)
                        demo_report[basename]={'declared_bytes':info.file_size,'embedding_dim_in_prefix':int(match.group(1)) if match else None,
                                               'prefix_keys':re.findall(r'"([^"\n]{1,60})"\s*:',prefix)[:12],
                                               'partial_header_only_crc_not_validated':True}
                    elif name.endswith('.graphml'):
                        with archive.open(info) as source: prefix=source.read(8192).decode('utf-8',errors='replace')
                        demo_report['graphml']={'declared_bytes':info.file_size,'key_names':re.findall(r'attr.name="([^"]+)"',prefix),
                                                'partial_header_only_crc_not_validated':True}
            else:
                types,pages,fields=Counter(),Counter(),Counter()
                documents=[]
                selected='commerce_manufacturing_0501317'
                for info in archive.infolist():
                    if info.filename.endswith('_content_list.json'):
                        records=json.loads(bounded_read(archive,info,2*1024**2))
                        types.update(record.get('type','unknown') for record in records)
                        fields.update(k for record in records for k in record)
                        page_ids={record.get('page_idx') for record in records if record.get('page_idx') is not None}
                        pages[len(page_ids)]+=1
                        documents.append({'document':info.filename.split('/')[1],'record_count':len(records),'observed_pages':len(page_ids)})
                    if not info.is_dir() and selected in info.filename and info.filename.endswith(('.md','.json','_origin.pdf')):
                        extract(archive,info,archive_id)
                (AUDIT/'dataset-structure.json').write_text(json.dumps({'parsed_document_count':len(documents),'types':dict(types),
                   'field_occurrences':dict(fields),'observed_page_total':sum(d['observed_pages'] for d in documents),
                   'page_count_distribution':dict(pages),'documents':documents,'selected_sample':selected,
                   'page_counts_are_content_list_observations_not_verified_PDF_page_counts':True},ensure_ascii=False,indent=2),encoding='utf-8')
    (AUDIT/'query-schema-only.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding='utf-8')
    (AUDIT/'index-structure.json').write_text(json.dumps(demo_report,ensure_ascii=False,indent=2),encoding='utf-8')
    (AUDIT/'extraction-manifest.json').write_text(json.dumps({'created_at_utc':datetime.now(timezone.utc).isoformat(),
       'total_extracted_bytes':total,'extracted_file_count':len(extractions),'files':extractions,
       'excluded':['pyc','editor databases','runtime environment files','gold query content','LLM response cache','large index bodies']},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'extracted_file_count':len(extractions),'bytes':total,'destination':str(DEST),'index_summary':demo_report,'query_schema_only':schema},ensure_ascii=False))


if __name__=='__main__': main()
