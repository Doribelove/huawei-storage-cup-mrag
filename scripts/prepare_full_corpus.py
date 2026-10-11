"""Freeze all100 source PDFs plus source-triggered native table overlays, without QA."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch
import zipfile
from evidence_lab.demo import EvidenceBrowser
from evidence_lab.native_pdf import native_tables,SourceVerifier
from evidence_lab.retrieval_policy import CandidatePolicy
from evidence_lab.safe_data import member_json,sha256_file
from evidence_lab.semantic import TOKENIZER_SHA256,token_windows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--phase-dir',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'full-manifest.json').exists() or (args.output/'regions.jsonl').exists():
        raise ValueError('Source freeze requires a new output directory')
    started=time.perf_counter();corpus=args.audit_dir/'regions.jsonl';before=sha256_file(corpus)
    with patch.object(socket.socket,'connect',side_effect=RuntimeError('Source freeze prohibits network')) as network:
        browser=EvidenceBrowser(args.audit_dir,args.phase_dir)
        browser.cache=args.output/'original-pdf-cache';browser.cache.mkdir(exist_ok=True)
        pages=defaultdict(set);overlays=[];failures=[];notes=[];parsed={}
        with zipfile.ZipFile(browser.data_path) as archive:
            for row in browser.records:
                if row['type']!='table': continue
                alias=next(a for a in row['aliases'] if a['document_name']==row['document_name'])
                member=alias['content_list_member']
                if member not in parsed: parsed[member]=member_json(archive,member)
                original=parsed[member][alias['record_index']]
                if not (original.get('table_body') or '').strip():
                    pages[row['document_name']].add(row['page'])
        for name in sorted(pages):
            pdf=browser.verified_pdf(name);doc=browser.documents[name];count=0
            for page in sorted(pages[name]):
                try:
                    native=native_tables(pdf,doc,name,page,browser.audit['source_archives']['data']['sha256'])
                    overlays.extend(native);count+=len(native)
                except ValueError as error:
                    failures.append({'document_name':name,'page':page,'reason':str(error)})
            note={'document_name':name,'source_gap_pages':sorted(pages[name]),'native_tables':count}
            notes.append(note);print(json.dumps(note),flush=True)
        verifier=SourceVerifier(browser,overlays)
        citations=[verifier.verify(row['evidence_id']) for row in overlays]
        if any(not v['all_passed'] for v in citations): raise ValueError('Native overlay source binding failed')
        rows=sorted(browser.records+overlays,key=lambda r:(r['document_name'],r['page'],r['evidence_id']))
        policy=CandidatePolicy(rows);searchable=policy.searchable()
        # Count actual token windows before launching a long source-embedding build.
        if sha256_file(args.model_dir/'tokenizer.json')!=TOKENIZER_SHA256: raise ValueError('Pinned tokenizer mismatch')
        import os
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False,use_fast=True)
        token_counts={};chunk_counts={}
        for row in searchable:
            ids=tokenizer(row['text'],add_special_tokens=False,truncation=False)['input_ids']
            if len(ids)>20000: raise ValueError('Source region token budget exceeded')
            token_counts[row['evidence_id']]=len(ids)
            chunk_counts[row['evidence_id']]=len(token_windows(len(ids),510,64))
        expected_chunks=sum(chunk_counts.values())
        if expected_chunks>20000: raise ValueError(f'Full corpus has {expected_chunks} chunks, requiring explicit index-budget revision')
        source=args.output/'regions.jsonl'
        source.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows),encoding='utf-8')
        native_path=args.output/'native-evidence.jsonl'
        native_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in overlays),encoding='utf-8')
        documents=[]
        for name,doc in sorted(browser.documents.items()):
            scoped=[r for r in rows if r['document_name']==name]
            eligible=[r for r in scoped if r['evidence_id'] in chunk_counts]
            documents.append({'name':name,'pdf_sha256':doc['sha256'],'pages':doc['page_count'],
                              'source_regions':len(scoped),'searchable_regions':len(eligible),
                              'expected_chunks':sum(chunk_counts[r['evidence_id']] for r in eligible),
                              'source_tokens':sum(token_counts[r['evidence_id']] for r in eligible),
                              'evidence_ids_sha256':hashlib.sha256(json.dumps(sorted(r['evidence_id'] for r in scoped)).encode()).hexdigest()})
        manifest={'version':'full-v1','source_corpus_sha256':before,'frozen_corpus_sha256':sha256_file(source),
                  'source_archive_sha256':browser.audit['source_archives']['data']['sha256'],
                  'documents':documents,'document_count':len(documents),'pdf_pages':sum(d['pages'] for d in documents),
                  'original_regions':len(browser.records),'native_overlay_regions':len(overlays),'source_regions':len(rows),
                  'expected_dense_chunks':expected_chunks,'token_window':{'max_length':512,'overlap':64},
                  'policy':policy.report(),'table_overlay_trigger':'Original parsed table_body is empty; no question/reference inputs',
                  'native_source_notes':notes,'native_extraction_failures':failures,'native_citations_all_passed':True,
                  'native_layer_sha256':sha256_file(native_path),'qa_questions_or_answers_read':False,
                  'source_corpus_unchanged':before==sha256_file(corpus),'socket_connect_attempts':network.call_count,
                  'preparation_seconds':round(time.perf_counter()-started,3),
                  'limits':['Source-only extraction; missing visual/scanned/borderless tables remain unresolved',
                            'Full corpus means100PDF coverage; geometric edge noise is not embedded',
                            'No answer or retrieval-quality metric measured in preparation']}
        if network.call_count or not manifest['source_corpus_unchanged'] or len(documents)!=100:
            raise ValueError('Full source freeze invariant failed')
        (args.output/'full-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (args.output/'native-citation-report.json').write_text(json.dumps(citations,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in manifest.items() if k not in ('documents','native_source_notes')},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
