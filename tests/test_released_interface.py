import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('released_adapter', PROJECT/'rag-anything-experiments/adapter.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class ReleasedInterfaceTests(unittest.TestCase):
    def fixture(self, root, name='previously_unseen_document'):
        from reportlab.pdfgen import canvas
        pdfs, parsed = root/'pdf', root/'parsed'
        pdfs.mkdir(); parsed.mkdir()
        c = canvas.Canvas(str(pdfs/(name+'.pdf')), pagesize=(300, 400))
        c.drawString(40, 300, 'Source evidence before evaluation.')
        c.save()
        (parsed/(name+'_content_list.json')).write_text(json.dumps([
            {'type': 'text', 'text': 'Source evidence before evaluation.', 'page_idx': 0, 'bbox': [100, 100, 800, 400]}]))
        return pdfs, parsed

    def test_new_document_build_has_recomputable_source_binding(self):
        from evidence_lab.portable import PortableSource
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdfs, parsed = self.fixture(root)
            result = adapter.import_documents(pdfs, parsed, root/'import')
            self.assertEqual(result['documents'], 1)
            self.assertEqual(result['identity_kind'], 'sha256_of_actual_member_manifest_not_a_zip_hash')
            source = PortableSource(root/'import')
            self.assertTrue(source.verify_citation(source.records[0]['evidence_id'])['all_passed'])
            self.assertEqual(set(source.documents), {'previously_unseen_document'})

    def test_unpaired_source_is_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdfs, parsed = self.fixture(root)
            (parsed/'extra_content_list.json').write_text('[]')
            with self.assertRaises(ValueError):
                adapter.import_documents(pdfs, parsed, root/'import')

    def test_ground_truth_field_cannot_be_imported_as_source_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdfs, parsed = self.fixture(root)
            path = next(parsed.glob('*.json'))
            value = adapter.read(path)
            value[0]['ground_truth'] = 'forbidden scorer field'
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                adapter.import_documents(pdfs, parsed, root/'import')

    def test_source_bytes_changed_after_import_are_rejected(self):
        from evidence_lab.portable import PortableSource
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdfs, parsed = self.fixture(root)
            adapter.import_documents(pdfs, parsed, root/'import')
            source = PortableSource(root/'import')
            source.verified_pdf('previously_unseen_document').write_bytes(b'tamper')
            with self.assertRaises(ValueError):
                source.verify_citation(source.records[0]['evidence_id'])

    def test_new_server_interface_question_only_and_refusal(self):
        class FixtureRunner:
            seen = []
            def answer(self, question):
                self.seen.append(question)
                return {'answer': '', 'status': 'withheld', 'query_time_s': 0.001}
        runner = FixtureRunner()
        server = HTTPServer(('127.0.0.1', 0), adapter.make_handler(runner))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = 'http://127.0.0.1:%d/query' % server.server_address[1]
            request = urllib.request.Request(endpoint, data=json.dumps({'question': 'Fixture question'}).encode(),
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=3) as response:
                value = json.load(response)
            self.assertEqual(value['answer'], '')
            self.assertEqual(runner.seen, ['Fixture question'])
            bad = urllib.request.Request(endpoint, data=json.dumps({'question': 'Q', 'answer': 'label'}).encode())
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(bad, timeout=3)
            self.assertEqual(caught.exception.code, 422)
            self.assertEqual(len(runner.seen), 1)
        finally:
            server.shutdown(); thread.join(); server.server_close()

    def test_no_overwrite_of_prior_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdfs, parsed = self.fixture(root)
            adapter.import_documents(pdfs, parsed, root/'import')
            with self.assertRaises(ValueError):
                adapter.import_documents(pdfs, parsed, root/'import')

if __name__ == '__main__':
    unittest.main()
