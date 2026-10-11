"""Freeze source-only sampling before pilot queries or model inference."""
import argparse
import hashlib
from collections import Counter
import json
from pathlib import Path
from evidence_lab.cli import load_evidence
from evidence_lab.pilot import select_pilot
from evidence_lab.safe_data import sha256_file

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--phase-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('configs/pilot-v1.manifest.json'))
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Frozen manifest exists; refusing overwrite')
    audit=json.loads((args.audit_dir/'data-audit.json').read_text(encoding='utf-8'))
    reviews=json.loads((args.phase_dir/'page-review.json').read_text(encoding='utf-8'))
    source=args.audit_dir/'regions.jsonl';corpus_hash=sha256_file(source)
    if corpus_hash!=audit['evidence_corpus_sha256']: raise ValueError('Corpus hash mismatch')
    rows=load_evidence(source);selected,rule=select_pilot(audit['dataset']['documents'],reviews)
    for doc in selected:
        scoped=[r for r in rows if r['document_name']==doc['name']]
        if any(r['doc_sha256']!=doc['sha256'] for r in scoped): raise ValueError('Source PDF binding mismatch')
        doc.update({'regions':len(scoped),'nonempty_regions':sum(bool(r['text'].strip()) for r in scoped),
            'region_types':dict(sorted(Counter(r['type'] for r in scoped).items())),
            'evidence_ids_sha256':hashlib.sha256(json.dumps(sorted(r['evidence_id'] for r in scoped)).encode()).hexdigest()})
    manifest={'role':'source_corpus_manifest','version':'pilot-v1','base_commit':'df31171d2b1edbe97b2cae085d87da6a2991d034',
        'source_corpus_sha256':corpus_hash,'source_archive_sha256':audit['source_archives']['data']['sha256'],
        'selection_rule':rule,'documents':selected,'source_only':True,'qa_gold_read':False,
        'scope':'10 source documents for development diagnostics; not official evaluation',
        'total_pages':sum(d['page_count'] for d in selected),'total_regions':sum(d['regions'] for d in selected),
        'total_nonempty_regions':sum(d['nonempty_regions'] for d in selected)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'manifest_sha256':sha256_file(args.output),**{k:v for k,v in manifest.items() if k not in ('documents','selection_rule')},
        'selected_documents':[(d['name'],d['page_count'],d['selection_slot']) for d in selected]},indent=2),flush=True)

if __name__=='__main__': main()
