from collections import defaultdict
from http.server import HTTPServer
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from evidence_lab.demo import EvidenceBrowser,make_handler
from evidence_lab.lexical import BM25


class DemoTests(unittest.TestCase):
    def test_explicit_relocated_archive_requires_frozen_hash_and_preserves_audit(self):
        import hashlib
        from pathlib import Path
        import tempfile
        import zipfile
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);audit=root/'audit';phase=root/'phase';audit.mkdir();phase.mkdir()
            archive=root/'relocated.zip'
            with zipfile.ZipFile(archive,'w') as zipped:zipped.writestr('source.txt','original fixture')
            (audit/'regions.jsonl').write_bytes(b'')
            metadata={'source_archives':{'data':{'path':str(root/'missing-original.zip'),
                'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}},
                'evidence_corpus_sha256':hashlib.sha256(b'').hexdigest(),'dataset':{'documents':{}}}
            raw=json.dumps(metadata).encode();(audit/'data-audit.json').write_bytes(raw)
            (phase/'page-review.json').write_text('[]',encoding='utf-8')
            (phase/'integrity-report.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(FileNotFoundError):EvidenceBrowser(audit,phase)
            cache=root/'separate-cache';browser=EvidenceBrowser(audit,phase,data_archive=archive,cache_dir=cache)
            self.assertEqual(browser.data_path,archive);self.assertEqual(browser.cache,cache)
            self.assertEqual((audit/'data-audit.json').read_bytes(),raw)
            self.assertFalse((phase/'demo-cache').exists())
            wrong=root/'wrong.zip'
            with zipfile.ZipFile(wrong,'w') as zipped:zipped.writestr('source.txt','different fixture')
            with self.assertRaisesRegex(ValueError,'Source archive changed'):EvidenceBrowser(audit,phase,data_archive=wrong,cache_dir=root/'bad-cache')
            self.assertFalse((root/'bad-cache').exists())

    def setUp(self):
        self.browser=EvidenceBrowser.__new__(EvidenceBrowser)
        self.browser.records=[{'evidence_id':'a','document_name':'alpha','page':1,'text':'pressure pressure'},
                              {'evidence_id':'b','document_name':'beta','page':1,'text':'pressure'}]
        for record in self.browser.records:
            record.update({'type':'text','source_bbox':[0,0,100,100],'bbox_normalized':[0,0,0.1,0.1]})
        self.browser.index=BM25(self.browser.records)
        self.browser.document_indexes={}
        self.browser.documents={name:{'sha256':name+'hash','page_count':1,
            'pages':[{'rotation':0,'cropbox':[0,0,600,800],'width':600,'height':800}]} for name in ('alpha','beta')}
        self.browser.anomalies=defaultdict(list)

    def test_selected_document_search_is_complete_and_has_no_answer(self):
        hits=self.browser.search('pressure','beta',1)
        self.assertEqual([row['evidence_id'] for row in hits['hits']],['b'])
        self.assertEqual(hits['score_scope'],'selected document')
        self.assertFalse(hits['answer_generated'])
        self.assertEqual(self.browser.search('absentword')['hits'],[])

    def test_invalid_scope_query_and_page_are_rejected(self):
        for call in (lambda:self.browser.search('pressure','missing'),lambda:self.browser.search('x'*501),
                     lambda:self.browser.search('pressure','beta',0),lambda:self.browser.checked_page('alpha',2),
                     lambda:self.browser.checked_page('../private',1)):
            with self.assertRaises(ValueError): call()

    def test_overlay_is_disabled_for_unverified_geometry(self):
        self.assertTrue(self.browser.page_info('alpha',1)['geometry_overlay_supported'])
        self.browser.documents['alpha']['pages'][0]['rotation']=90
        self.assertFalse(self.browser.page_info('alpha',1)['geometry_overlay_supported'])
        self.browser.documents['alpha']['pages'][0]['rotation']=0
        self.browser.documents['alpha']['pages'][0]['cropbox']=[20,0,600,800]
        self.assertFalse(self.browser.page_info('alpha',1)['geometry_overlay_supported'])

    def test_http_search_and_rejection_use_the_real_handler(self):
        server=HTTPServer(('127.0.0.1',0),make_handler(self.browser))
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base+'/api/search?q=pressure&doc=beta&k=1',timeout=5) as response:
                result=json.load(response)
                self.assertEqual(result['hits'][0]['document_name'],'beta')
                self.assertIn("default-src 'self'",response.headers['Content-Security-Policy'])
            for path,status in (('/api/search?q=pressure&doc=missing',400),('/private-file',404)):
                with self.subTest(path=path),self.assertRaises(HTTPError) as error: urlopen(base+path,timeout=5)
                self.assertEqual(error.exception.code,status)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=5)


if __name__=='__main__': unittest.main()
