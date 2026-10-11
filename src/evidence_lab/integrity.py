"""Derived source links and explicit page fallbacks; original indexes stay untouched."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile

from .cli import load_evidence
from .safe_data import document_name,member_json,preflight_zip,read_member,reject_answer_fields,sha256_file,validate_name


def link_image_chunk(chunk:dict,evidence:list[dict],documents:dict,asset_hash)->dict:
    """An exact source asset link proves provenance, not historical VLM assertions."""
    reject_answer_fields(chunk)
    result={'chunk_id':chunk['__id__'],'original_full_doc_id':chunk.get('full_doc_id'),
            'original_file_path':chunk.get('file_path'),'link_status':'unknown',
            'content_verification':'unknown','historical_analysis_usable_as_verified_evidence':False,
            'source_links':[]}
    name=document_name(chunk.get('file_path',''))
    doc=documents.get(name)
    if not doc:
        result['reason']='Referenced original PDF is not in verified dataset'
        return result
    paths=re.findall(r'^Image Path:\s*([^\r\n]+)',chunk.get('content',''),re.MULTILINE)
    if len(paths)!=1:
        result['reason']='No unique explicit image reference; no inferred cross-document link'
        return result
    try: path=validate_name(paths[0].strip())
    except ValueError:
        result['reason']='Unsafe source asset path'
        return result
    candidates=[r for r in evidence if r.get('document_name')==name and r.get('doc_sha256')==doc['sha256']
                and r.get('image_member')==path and any(a.get('pdf_member')==doc['member'] for a in r.get('aliases',[]))]
    if len(candidates)!=1:
        result['reason']='Image reference does not identify one region in the target original PDF'
        return result
    region=candidates[0]
    try: digest=asset_hash(path)
    except (KeyError,ValueError):
        result['reason']='Referenced source asset cannot be verified'
        return result
    captions=re.findall(r'^Captions:\s*([^\r\n]*)',chunk.get('content',''),re.MULTILINE)
    normalize=lambda text:re.sub(r'\s+',' ',text).strip()
    result.update({'link_status':'asset_linked','reason':'Exact same-document image member and verified PDF-backed region',
        'source_links':[{'evidence_id':region['evidence_id'],'document_name':name,'doc_sha256':doc['sha256'],
                         'pdf_member':doc['member'],'page':region['page'],'source_bbox':region['source_bbox'],
                         'image_member':path,'image_sha256':digest,
                         'caption_exact_after_whitespace_normalization':len(captions)==1 and normalize(captions[0])==normalize(region['text'])}]})
    return result


def page_fallback(review:dict)->dict:
    if review.get('visual_classification') not in {'blank','scan','parse-missing','uncertain'}:
        raise ValueError('Page must have an explicit supported review classification')
    digest=review['doc_sha256']
    page=review['page']
    if not isinstance(page,int) or isinstance(page,bool) or page<1: raise ValueError('Invalid fallback page')
    return {'role':'source_pdf_page','evidence_id':hashlib.sha256(f'{digest}:{page}:whole-page'.encode()).hexdigest(),
            'document_name':review['document_name'],'doc_sha256':digest,'page':page,
            'pdf_member':review['pdf_member'],'bbox_normalized':[0,0,1,1],
            'representation':'whole_page_visual_fallback','text':'','automatically_searchable':False,
            'classification':review['visual_classification'],'review_note':review['review_note'],
            'render_path':review['render_path'],'render_sha256':review['render_sha256'],
            'source_anomalies':review['anomalies'],
            'limits':['No guessed table cells or OCR text','Invalid original bboxes remain quarantined and unchanged',
                      'Blank means visually blank at the recorded rendering resolution']}


def build_integrity(audit_dir:Path,page_reviews:Path,output:Path)->dict:
    audit=json.loads((audit_dir/'data-audit.json').read_text(encoding='utf-8'))
    index_path=audit_dir/'vdb_chunks.snapshot.json'
    before=sha256_file(index_path)
    if before!=audit['chunk_snapshot']['sha256']: raise ValueError('Original index snapshot hash mismatch')
    if sha256_file(audit_dir/'regions.jsonl')!=audit['evidence_corpus_sha256']: raise ValueError('Evidence corpus hash mismatch')
    data=Path(audit['source_archives']['data']['path'])
    demo=Path(audit['source_archives']['demo']['path'])
    for name,path in (('data',data),('demo',demo)):
        preflight_zip(path)
        if sha256_file(path)!=audit['source_archives'][name]['sha256']: raise ValueError('Original archive changed')
    if index_path.stat().st_size>128*1024**2: raise ValueError('Index exceeds bounded JSON budget')
    storage=json.loads(index_path.read_text(encoding='utf-8'))
    evidence=load_evidence(audit_dir/'regions.jsonl')
    with zipfile.ZipFile(demo) as archive:
        member=next(n for n in archive.namelist() if n.endswith('/kv_store_doc_status.json'))
        statuses=member_json(archive,member)
    orphans=[row for row in storage['data'] if row.get('full_doc_id') not in statuses]
    with zipfile.ZipFile(data) as archive:
        cache={}
        def asset_hash(member):
            if member not in cache: cache[member]=hashlib.sha256(read_member(archive,member,16*1024**2)).hexdigest()
            return cache[member]
        links=[link_image_chunk(row,evidence,audit['dataset']['documents'],asset_hash) for row in orphans]
    reviews=json.loads(page_reviews.read_text(encoding='utf-8'))
    fallbacks=[]
    for review in reviews:
        doc=audit['dataset']['documents'].get(review['document_name'])
        if not doc or review['doc_sha256']!=doc['sha256'] or review['pdf_member']!=doc['member']:
            raise ValueError('Page review provenance mismatch')
        if not 1<=review['page']<=doc['page_count']: raise ValueError('Page review outside original PDF')
        render=Path(review['render_path']).resolve()
        if not render.is_relative_to(output.resolve()): raise ValueError('Review image outside isolated output')
        if sha256_file(render)!=review['render_sha256']: raise ValueError('Reviewed page render changed')
        fallbacks.append(page_fallback(review))
    if sha256_file(index_path)!=before: raise RuntimeError('Original index changed')
    report={'kind':'derived integrity layer; not an index repair or accuracy evaluation',
            'source_index_sha256':before,'source_index_unchanged':True,'source_archives_unchanged':True,
            'source_corpus_sha256':audit['evidence_corpus_sha256'],'source_status_not_rewritten':True,
            'orphan_chunks':len(orphans),'link_status_counts':dict(Counter(r['link_status'] for r in links)),
            'verified_historical_analysis_count':0,'links':links,
            'page_review_count':len(reviews),'page_classifications':dict(Counter(r['visual_classification'] for r in reviews)),
            'fallback_count':len(fallbacks),'model_calls':0,'network_calls':0}
    output.mkdir(parents=True,exist_ok=True)
    (output/'integrity-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (output/'page-fallbacks.jsonl').open('w',encoding='utf-8') as file:
        for row in fallbacks: file.write(json.dumps(row,ensure_ascii=False)+'\n')
    return report


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,default=Path('output/audit'))
    parser.add_argument('--page-reviews',type=Path,default=Path('output/audit/phase2/page-review.json'))
    parser.add_argument('--output',type=Path,default=Path('output/audit/phase2'))
    args=parser.parse_args()
    report=build_integrity(args.audit_dir,args.page_reviews,args.output)
    print(json.dumps({key:value for key,value in report.items() if key!='links'},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
