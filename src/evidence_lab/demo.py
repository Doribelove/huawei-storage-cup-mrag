"""Loopback-only evidence browser. No model, remote API, or document tool execution."""
from __future__ import annotations

import argparse
from collections import defaultdict
from http.server import BaseHTTPRequestHandler,HTTPServer
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import parse_qs,urlsplit
import zipfile

from .cli import load_evidence
from .lexical import BM25
from .safe_data import member_json,normalized_bbox,preflight_zip,read_member,record_text,sha256_file


class EvidenceBrowser:
    def __init__(self,audit_dir:Path,phase_dir:Path,*,data_archive:Path|None=None,cache_dir:Path|None=None):
        self.audit=json.loads((audit_dir/'data-audit.json').read_text(encoding='utf-8'))
        # An explicit relocation is accepted only after the original archive hash
        # checks below; the frozen audit/provenance record is never rewritten.
        self.data_path=Path(data_archive) if data_archive is not None else Path(self.audit['source_archives']['data']['path'])
        preflight_zip(self.data_path)
        if sha256_file(self.data_path)!=self.audit['source_archives']['data']['sha256']:
            raise ValueError('Source archive changed')
        if sha256_file(audit_dir/'regions.jsonl')!=self.audit['evidence_corpus_sha256']:
            raise ValueError('Evidence corpus changed')
        self.records=load_evidence(audit_dir/'regions.jsonl')
        self.byid={r['evidence_id']:r for r in self.records}
        self.index=BM25(self.records)
        self.document_indexes={}
        self.documents=self.audit['dataset']['documents']
        self.reviews=json.loads((phase_dir/'page-review.json').read_text(encoding='utf-8'))
        self.integrity=json.loads((phase_dir/'integrity-report.json').read_text(encoding='utf-8'))
        self.anomalies=defaultdict(list)
        for review in self.reviews: self.anomalies[(review['document_name'],review['page'])].append(review)
        self.cache=Path(cache_dir) if cache_dir is not None else phase_dir/'demo-cache'
        self.cache.mkdir(parents=True,exist_ok=True)
        self.assets=Path(__file__).parent/'web'

    def metadata(self)->dict:
        return {'documents':[{'name':name,'pages':doc['page_count']} for name,doc in sorted(self.documents.items())],
            'regions':len(self.records),'reviewed_pages':len(self.reviews),
            'page_classifications':self.integrity['page_classifications'],
            'orphan_chunks':self.integrity['orphan_chunks'],'asset_links':self.integrity['link_status_counts'],
            'source_index_unchanged':self.integrity['source_index_unchanged'],
            'source_index_sha256':self.integrity['source_index_sha256'],
            'model_calls':0,'boundary':'BM25证据浏览；未接入bge-m3/Qwen；历史图像分析内容未验证。'}

    def checked_page(self,name:str,page:int)->dict:
        if name not in self.documents or isinstance(page,bool) or not isinstance(page,int): raise ValueError('Unknown document/page')
        doc=self.documents[name]
        if not 1<=page<=doc['page_count']: raise ValueError('Page outside verified original PDF')
        return doc

    def page_info(self,name:str,page:int)->dict:
        doc=self.checked_page(name,page)
        regions=[{'evidence_id':r['evidence_id'],'type':r['type'],'source_bbox':r['source_bbox'],
                  'bbox_normalized':r['bbox_normalized'],'text':r['text']} for r in self.records
                 if r['document_name']==name and r['page']==page]
        reviews=self.anomalies.get((name,page),[])
        return {'document_name':name,'page':page,'doc_sha256':doc['sha256'],'regions':regions,
                'anomalies':[{'classification':r['visual_classification'],'note':r['review_note'],
                    'repair_suggestion':r['repair_suggestion'],'items':r['anomalies']} for r in reviews],
                'geometry_overlay_supported':doc['pages'][page-1]['rotation']==0 and doc['pages'][page-1]['cropbox']==
                    [0.0,0.0,doc['pages'][page-1]['width'],doc['pages'][page-1]['height']],
                'geometry_note':'区域坐标为MinerU量化估计；异常原坐标保留，未裁剪或补猜内容。'}

    def search(self,query:str,document:str='',top_k:int=5)->dict:
        if not isinstance(query,str) or len(query)>500: raise ValueError('Query exceeds 500 characters')
        if document and document not in self.documents: raise ValueError('Unknown document filter')
        if not 1<=top_k<=100: raise ValueError('top_k must be 1..100')
        start=time.perf_counter()
        if document and document not in self.document_indexes:
            self.document_indexes[document]=BM25([r for r in self.records if r['document_name']==document])
        index=self.document_indexes[document] if document else self.index
        hits=index.search(query,top_k)
        for hit in hits:
            hit['anomaly_classes']=[r['visual_classification'] for r in self.anomalies.get((hit['document_name'],hit['page']),[])]
        return {'query':query,'document_filter':document,'hits':hits,'latency_ms':round((time.perf_counter()-start)*1000,3),
                'model_calls':0,'answer_generated':False,
                'score_scope':'selected document' if document else 'all source documents'}

    def verified_pdf(self,name:str)->Path:
        doc=self.documents[name]
        target=self.cache/(name+'.pdf')
        if target.exists():
            if sha256_file(target)!=doc['sha256']: raise ValueError('Cached PDF hash mismatch')
            return target
        with zipfile.ZipFile(self.data_path) as archive: raw=read_member(archive,doc['member'],32*1024**2)
        if hashlib.sha256(raw).hexdigest()!=doc['sha256']: raise ValueError('PDF member hash mismatch')
        target.write_bytes(raw)
        return target

    def rendered_page(self,name:str,page:int)->Path:
        self.checked_page(name,page)
        target=self.cache/f'{name}-p{page}.png'
        if not target.exists():
            executable=shutil.which('pdftoppm')
            if not executable: raise ValueError('Poppler pdftoppm is unavailable')
            subprocess.run([executable,'-f',str(page),'-l',str(page),'-singlefile','-r','100','-png',
                            str(self.verified_pdf(name)),str(target.with_suffix(''))],
                           check=True,capture_output=True,timeout=30)
        return target

    def verify_citation(self,evidence_id:str)->dict:
        row=self.byid.get(evidence_id)
        if not row: raise ValueError('Unknown evidence ID')
        doc=self.checked_page(row['document_name'],row['page'])
        alias=next(a for a in row['aliases'] if a['document_name']==row['document_name'])
        with zipfile.ZipFile(self.data_path) as archive:
            parsed=member_json(archive,alias['content_list_member'])
            source=parsed[alias['record_index']]
            checks={'pdf_sha256':sha256_file(self.verified_pdf(row['document_name']))==row['doc_sha256'],
                    'content_list_sha256':hashlib.sha256(read_member(archive,alias['content_list_member'],2*1024**2)).hexdigest()==doc['content_list']['sha256'],
                    'page':source['page_idx']+1==row['page'],'source_bbox':source['bbox']==row['source_bbox'],
                    'normalized_bbox':normalized_bbox(source)==row['bbox_normalized'],'text':record_text(source)==row['text']}
        return {'evidence_id':evidence_id,'checks':checks,'all_passed':all(checks.values()),
                'meaning':'原始PDF与解析记录绑定核验；不是答案正确性判断。'}


def make_handler(browser:EvidenceBrowser):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_): pass

        def send_bytes(self,data:bytes,content_type:str,status:int=200):
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self'; connect-src 'self'; frame-src 'self'; object-src 'none'")
            self.end_headers()
            self.wfile.write(data)

        def json(self,value,status=200): self.send_bytes(json.dumps(value,ensure_ascii=False).encode('utf-8'),'application/json; charset=utf-8',status)

        def do_GET(self):
            url=urlsplit(self.path)
            args=parse_qs(url.query)
            arg=lambda key,default='':args.get(key,[default])[0]
            try:
                if url.path in ('/','/app.js','/style.css'):
                    name={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}[url.path]
                    mime={'index.html':'text/html','app.js':'text/javascript','style.css':'text/css'}[name]
                    self.send_bytes((browser.assets/name).read_bytes(),mime+'; charset=utf-8')
                elif url.path=='/api/meta': self.json(browser.metadata())
                elif url.path=='/api/search':
                    if hasattr(browser,'search_with_options'):
                        self.json(browser.search_with_options(arg('q'),arg('doc'),int(arg('k','5')),arg('mode','hybrid')))
                    else: self.json(browser.search(arg('q'),arg('doc'),int(arg('k','5'))))
                elif url.path=='/api/page': self.json(browser.page_info(arg('doc'),int(arg('page'))))
                elif url.path=='/api/verify': self.json(browser.verify_citation(arg('id')))
                elif url.path=='/api/anomalies': self.json({'pages':browser.reviews,'links':browser.integrity['links']})
                elif url.path=='/page.png': self.send_bytes(browser.rendered_page(arg('doc'),int(arg('page'))).read_bytes(),'image/png')
                else: self.json({'error':'Unknown local endpoint'},404)
            except (ValueError,KeyError,StopIteration): self.json({'error':'Invalid query or unavailable evidence'},400)
            except (OSError,subprocess.SubprocessError): self.json({'error':'Local rendering or file read failed'},500)
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,default=Path('output/audit'))
    parser.add_argument('--phase-dir',type=Path,default=Path('output/audit/phase2'))
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--data-archive',type=Path,help='Explicit relocated original archive; exact frozen SHA required')
    parser.add_argument('--cache-dir',type=Path,help='Task-owned writable page cache; source audit stays unchanged')
    args=parser.parse_args()
    browser=EvidenceBrowser(args.audit_dir,args.phase_dir,data_archive=args.data_archive,cache_dir=args.cache_dir)
    server=HTTPServer(('127.0.0.1',args.port),make_handler(browser))
    print(f'Local evidence browser: http://127.0.0.1:{server.server_port} (Ctrl+C to stop)',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__=='__main__': main()
