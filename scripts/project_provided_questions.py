"""One-time isolated QA projection: runtime questions and scorer-only labels stay separate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import zipfile
from evidence_lab.evaluation import load_questions,reference_pages
from evidence_lab.safe_data import preflight_zip,read_member,sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()): raise ValueError('Question projection needs an empty new directory')
    audit=json.loads((args.audit_dir/'data-audit.json').read_text(encoding='utf-8'))
    archive_path=Path(audit['source_archives']['demo']['path']);preflight_zip(archive_path)
    if sha256_file(archive_path)!=audit['source_archives']['demo']['sha256']: raise ValueError('Provided archive mismatch')
    quarantine=audit['query_quarantine']
    with zipfile.ZipFile(archive_path) as archive: raw=read_member(archive,quarantine['member'],2*1024**2)
    import hashlib
    if hashlib.sha256(raw).hexdigest()!=quarantine['sha256']: raise ValueError('Provided query file mismatch')
    records=json.loads(raw.decode('utf-8-sig'));questions=[];labels=[]
    for record in records:
        questions.append({key:record[key] for key in ('sample_id','question')})
        pages,alignment=reference_pages(record,audit['dataset']['documents'])
        labels.append({'sample_id':record['sample_id'],'reference_pages':pages,'alignment':alignment,
                       'domain':record['domain'],'answer_type':record['answer_type'],'question_type':record['question_type'],
                       'answer':record['answer']})
    qpath=args.output/'questions.json';lpath=args.output/'scorer-only-labels.json'
    qpath.write_text(json.dumps({'version':'question-only-v1','questions':questions},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if len(load_questions(qpath))!=30: raise ValueError('Expected all30 provided questions')
    lpath.write_text(json.dumps(labels,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest={'version':'provided30-v1','source_query_sha256':quarantine['sha256'],
              'questions_sha256':sha256_file(qpath),'scorer_only_labels_sha256':sha256_file(lpath),'sample_count':30,
              'runtime_fields':['sample_id','question'],'runtime_reference_or_answer_fields':False,
              'answer_access':'Only this isolated projection process and later independent scorer; never source index/retriever/reader',
              'page_reference_alignment':'Filename numbering established against complete one-based original PDF page sets',
              'freeze_before_first_full_retrieval':True,'limitations':['Supplied30-example evaluation; no official blind-test or award claim']}
    (args.output/'evaluation-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__': main()
