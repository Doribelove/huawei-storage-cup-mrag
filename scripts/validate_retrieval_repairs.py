"""Real-source and genuine BGE regressions; no QA answers or claimed accuracy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch
from evidence_lab.cli import load_evidence
from evidence_lab.demo import EvidenceBrowser
from evidence_lab.lexical import BM25
from evidence_lab.native_pdf import native_tables,SourceVerifier
from evidence_lab.pilot import ContextAssembler
from evidence_lab.retrieval_policy import CandidatePolicy
from evidence_lab.safe_data import sha256_file
from evidence_lab.semantic import LocalBGE,DenseIndex


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--phase-dir',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--pilot-dir',type=Path,default=Path('output/pilot-v1'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    target=args.output/'retrieval-repairs.json'
    if target.exists(): raise ValueError('Choose a fresh output directory; original results preserved')
    source=args.audit_dir/'regions.jsonl';before=sha256_file(source)
    with patch.object(socket.socket,'connect',side_effect=RuntimeError('Offline diagnostic')) as network:
        browser=EvidenceBrowser(args.audit_dir,args.phase_dir)
        browser.cache=args.output/'citation-cache';browser.cache.mkdir(exist_ok=True)
        name='crm_4502338';doc=browser.documents[name]
        native=native_tables(browser.verified_pdf(name),doc,name,8,browser.audit['source_archives']['data']['sha256'])
        old=[r for r in browser.records if r['document_name']==name]
        legacy_hits=BM25(old).search('Receiving Clerk',5)
        repaired_hits=BM25(CandidatePolicy(old+native).searchable()).search('Receiving Clerk',5)
        verifier=SourceVerifier(browser,native)
        verified=verifier.verify(repaired_hits[0]['evidence_id'])
        if repaired_hits[0]['page']!=8 or not verified['all_passed']:
            raise ValueError('Original-table retrieval or provenance repair failed')
        (args.output/'native-evidence.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in native),encoding='utf-8')
        manifest=json.loads(Path('configs/pilot-v1.manifest.json').read_text(encoding='utf-8'))
        rows=[]
        for document in manifest['documents']:
            rows.extend(sorted([r for r in browser.records if r['document_name']==document['name']],key=lambda r:r['evidence_id']))
        pilot=json.loads((args.pilot_dir/'pilot-report.json').read_text(encoding='utf-8'))
        index_path=args.pilot_dir/'pilot.nano.json'
        if sha256_file(index_path)!=pilot['index_sha256']: raise ValueError('Reviewed pilot index mismatch')
        model=LocalBGE(args.model_dir)
        vector,query_stats=model.query('zzzxqvnonev290')
        index=DenseIndex(index_path,rows,model.identity,source_tokenizer=model.tokenizer)
        policy=CandidatePolicy(rows)
        dense_before=index.search(vector,model.identity,5)
        dense_after=index.search(vector,model.identity,5,candidate_filter=policy.accepts)
        contexts=ContextAssembler(rows).assemble(dense_after,candidate_filter=policy.accepts)
        if any(not policy.accepts(r) for r in dense_after+contexts['added_regions']):
            raise ValueError('Filtered noise re-entered context')
        note=lambda hits:[{'evidence_id':h['evidence_id'],'document_name':h['document_name'],'page':h['page'],
                          'source_kind':h.get('extraction',{}).get('kind','mineru'),'text':h['text']} for h in hits]
        report={'kind':'Real original-source and genuine local BGE regression; not answer accuracy',
                'table_gap':{'query':'Receiving Clerk','before':note(legacy_hits),'after':note(repaired_hits),
                             'citation':verified,'native_regions':len(native)},
                'nonsense':{'query':'zzzxqvnonev290','before':note(dense_before),'after':note(dense_after),
                            'filtered_page_number_hits_before':sum(policy.excluded.get(h['evidence_id'])=='page_edge_number' for h in dense_before),
                            'filtered_noise_hits_after':sum(h['evidence_id'] in policy.excluded for h in dense_after),
                            'context':contexts,'query_stats':query_stats,'model_identity':model.identity},
                'policy':policy.report(),'socket_connect_attempts':network.call_count,
                'source_corpus_sha256':before,'source_corpus_unchanged':before==sha256_file(source),
                'reviewed_pilot_index_unchanged':sha256_file(index_path)==pilot['index_sha256'],
                'qa_answers_read':False,'supported_answer':False,'answer_generated':False,
                'limitations':['Nonsense still yields semantic candidates; no calibrated answerability claim',
                               'Native tables require embedded PDF text and lines; scanned tables need the reader',
                               'Two selected failure regressions are development results, not a held-out metric']}
        if network.call_count or not report['source_corpus_unchanged'] or not report['reviewed_pilot_index_unchanged']:
            raise ValueError('Offline/source preservation invariant failed')
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'native_table_regions':len(native),'old_first_page':legacy_hits[0]['page'],
                          'new_first_page':repaired_hits[0]['page'],'citation_all_passed':verified['all_passed'],
                          'old_footer_number_hits':report['nonsense']['filtered_page_number_hits_before'],
                          'new_noise_hits':report['nonsense']['filtered_noise_hits_after'],
                          'socket_connect_attempts':network.call_count,'source_corpus_unchanged':report['source_corpus_unchanged'],
                          'answer_accuracy_measured':False},indent=2),flush=True)


if __name__=='__main__': main()
