"""Handoff-only extracted-member resolver. Does not modify product source or frozen audit.

Original ZIP SHA was checked before extraction on Lenovo. This resolver verifies the
original PDF/content-list member hashes from the immutable audit at use time; it
does not assert that an absent whole ZIP was rehashed on AGX.
"""
import hashlib, json
from pathlib import Path
from evidence_lab.cli import load_evidence
from evidence_lab.demo import EvidenceBrowser
from evidence_lab.safe_data import sha256_file, normalized_bbox, record_text

class PortableSource(EvidenceBrowser):
    def __init__(self, root):
        self.root=Path(root).resolve()
        self.audit=json.loads((self.root/'data/audit/data-audit.json').read_text('utf-8'))
        original=self.root/'data/audit/regions.jsonl'
        if sha256_file(original)!=self.audit['evidence_corpus_sha256']:
            raise ValueError('Original corpus changed')
        self.records=load_evidence(original);self.byid={r['evidence_id']:r for r in self.records}
        self.documents=self.audit['dataset']['documents']
        mapping=json.loads((self.root/'data/source-members.json').read_text('utf-8'))
        if mapping['original_archive_sha256']!=self.audit['source_archives']['data']['sha256']:
            raise ValueError('Archive provenance mismatch')
        self.members={m['original_member']:m for m in mapping['members']}
        if len(self.members)!=len(mapping['members']):raise ValueError('Duplicate member')
    def member(self,name,expected_sha):
        item=self.members.get(name)
        if item is None: raise ValueError('Required source member not in smoke bundle: '+name)
        p=(self.root/item['relative_path']).resolve()
        if not p.is_relative_to(self.root) or item['sha256']!=expected_sha or not p.is_file():
            raise ValueError('Invalid member mapping')
        if p.stat().st_size!=item['bytes'] or sha256_file(p)!=expected_sha:
            raise ValueError('Source member bytes changed')
        return p
    def verified_pdf(self,name):
        doc=self.documents[name]
        return self.member(doc['member'],doc['sha256'])
    def verify_citation(self,evidence_id):
        row=self.byid.get(evidence_id)
        if row is None: raise ValueError('Unknown original evidence ID')
        doc=self.checked_page(row['document_name'],row['page'])
        alias=next(a for a in row['aliases'] if a['document_name']==row['document_name'])
        raw=self.member(alias['content_list_member'],doc['content_list']['sha256']).read_bytes()
        source=json.loads(raw)[alias['record_index']]
        checks={'pdf_sha256':sha256_file(self.verified_pdf(row['document_name']))==row['doc_sha256'],
            'content_list_sha256':hashlib.sha256(raw).hexdigest()==doc['content_list']['sha256'],
            'page':source['page_idx']+1==row['page'],'source_bbox':source['bbox']==row['source_bbox'],
            'normalized_bbox':normalized_bbox(source)==row['bbox_normalized'],'text':record_text(source)==row['text']}
        return {'evidence_id':evidence_id,'checks':checks,'all_passed':all(checks.values()),
            'meaning':'Original extracted PDF/parsed-member binding; no whole-archive rehash on receiver, no answer correctness'}
