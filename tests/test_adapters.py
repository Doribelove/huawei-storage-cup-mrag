import base64
from io import BytesIO
import json
import math
from pathlib import Path
import socket
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import zlib

import numpy as np

from evidence_lab.cli import load_evidence
from evidence_lab.lexical import BM25
from evidence_lab.material_audit import audit_archives,reconcile_status
from evidence_lab.nano_adapter import ReadOnlyNanoIndex,decode_cached_vector
from evidence_lab.safe_data import image_member,normalized_bbox,preflight_zip,read_member,record_text,validate_name
from evidence_lab.score import project_score


class SafetyTests(unittest.TestCase):
    def test_windows_and_traversal_paths(self):
        for path in ('../escape.txt','safe/../../escape.txt','C:/escape.txt','/escape','file:stream','NUL.txt','a./b','a /b'):
            with self.subTest(path=path),self.assertRaises(ValueError): validate_name(path)

    def test_read_budget_and_crc(self):
        data=BytesIO()
        with zipfile.ZipFile(data,'w') as archive: archive.writestr('file.txt',b'content')
        data.seek(0)
        with zipfile.ZipFile(data) as archive:
            with self.assertRaises(ValueError): read_member(archive,'file.txt',3)
            self.assertEqual(read_member(archive,'file.txt',10),b'content')

    def test_symlink_rejected(self):
        data=BytesIO()
        info=zipfile.ZipInfo('link')
        info.create_system=3
        info.external_attr=0o120777<<16
        with zipfile.ZipFile(data,'w') as archive: archive.writestr(info,'../outside')
        data.seek(0)
        with zipfile.ZipFile(data) as archive,self.assertRaises(ValueError): read_member(archive,'link',100)

    def test_gold_cannot_become_source(self):
        with self.assertRaises(ValueError): record_text({'type':'text','text':'good','answer':'hidden'})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'gold.jsonl'
            path.write_text(json.dumps({'role':'source_pdf_region','answer':'hidden'})+'\n')
            with self.assertRaises(ValueError): load_evidence(path)

    def test_bbox_checks(self):
        self.assertEqual(normalized_bbox({'bbox':[0,100,1000,900]}),[0,0.1,1,0.9])
        for bbox in ([0,0,1001,900],[5,0,1,1],[0,0,float('nan'),1]):
            with self.assertRaises(ValueError): normalized_bbox({'bbox':bbox})

    def test_external_image_is_never_opened(self):
        member='mineru-parsed/doc/auto/doc_content_list.json'
        self.assertIsNone(image_member({'img_path':'C:/private/secret.jpg'},member,set()))
        self.assertEqual(image_member({'img_path':'/old/machine/a.jpg'},member,{'mineru-parsed/doc/auto/images/a.jpg'}),'mineru-parsed/doc/auto/images/a.jpg')

    def test_duplicate_chain_is_not_two_missing_documents(self):
        status={'a':{'file_path':'commerce_x.pdf','status':'processed'},
            'b':{'file_path':'education_x.pdf','status':'failed','metadata':{'is_duplicate':True,'original_doc_id':'a'}},
            'c':{'file_path':'education_x.pdf','status':'failed','metadata':{'is_duplicate':True,'original_doc_id':'b'}}}
        result=reconcile_status(status,{'commerce_x','education_x'})
        self.assertEqual(result['unique_file_name_count'],2)
        self.assertEqual(result['unresolved_after_status_aliases'],[])
        status['a']['metadata']={'original_doc_id':'c'}
        with self.assertRaises(ValueError): reconcile_status(status,{'commerce_x','education_x'})


class RetrievalTests(unittest.TestCase):
    def test_bm25_answerless_and_empty_result(self):
        records=[{'evidence_id':'1','text':'Hydraulic pressure limit is twelve kilopascals'},
                 {'evidence_id':'2','text':'Temperature limit is twenty Celsius'}]
        index=BM25(records)
        self.assertEqual(index.search('hydraulic pressure',1)[0]['evidence_id'],'1')
        self.assertEqual(index.search('warranty years'),[])
        self.assertEqual(BM25([]).search('pressure'),[])

    def test_score_caps_and_negative_penalties(self):
        score=project_score(0.9,100,100,1,1)
        self.assertEqual(score['projected_score'],84)
        self.assertEqual(project_score(0.9,2,0.5,1,1)['projected_score'],91.5)
        with self.assertRaises(ValueError): project_score(0.9,1,1,0,1)

    def test_bounded_cached_vector(self):
        vector=np.ones(1024,dtype=np.float16)
        encoded=base64.b64encode(zlib.compress(vector.tobytes())).decode()
        self.assertEqual(decode_cached_vector(encoded).shape,(1024,))
        bomb=base64.b64encode(zlib.compress(b'X'*1000000)).decode()
        with self.assertRaises(ValueError): decode_cached_vector(bomb)
        with self.assertRaises(ValueError): decode_cached_vector('invalid%%')

    def test_real_nano_reads_without_network_or_writes(self):
        matrix=np.zeros((3,1024),dtype=np.float32)
        matrix[0,0]=1
        matrix[1,1]=1
        matrix[2,:2]=[0.8,0.6]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'index.json'
            storage={'embedding_dim':1024,'data':[{'__id__':str(i),'content':'numeric database test fixture'} for i in range(3)],
                     'matrix':base64.b64encode(matrix.tobytes()).decode()}
            path.write_text(json.dumps(storage),encoding='utf-8')
            with patch.object(socket.socket,'connect',side_effect=AssertionError('Network prohibited')):
                index=ReadOnlyNanoIndex(path)
                self.assertEqual(index.query_vector(matrix[0],1)[0]['id'],'0')
                self.assertTrue(index.replay_record(1)['self_in_top_k'])
                with self.assertRaises(ValueError): index.query_vector(np.ones(3))
                with self.assertRaises(ValueError): index.query_vector(np.zeros(1024))
                index.assert_unchanged()
            storage['embedding_dim']=1536
            path.write_text(json.dumps(storage),encoding='utf-8')
            with self.assertRaises(ValueError): ReadOnlyNanoIndex(path)


class ReviewRegressionTests(unittest.TestCase):
    def test_gt_answer_rejected_by_all_evidence_entrypoints(self):
        record={'role':'source_pdf_region','evidence_id':'x','text':'pressure','gt_answer':'SYNTHETIC_GOLD'}
        with self.assertRaises(ValueError): record_text(record)
        with self.assertRaises(ValueError): BM25([record])
        with tempfile.TemporaryDirectory() as directory:
            corpus=Path(directory)/'corpus.jsonl'
            corpus.write_text(json.dumps(record)+'\n',encoding='utf-8')
            with self.assertRaises(ValueError): load_evidence(corpus)
            matrix=np.zeros((1,1024),dtype=np.float32)
            matrix[0,0]=1
            path=Path(directory)/'index.json'
            storage={'embedding_dim':1024,'data':[{'__id__':'x',**record}],
                     'matrix':base64.b64encode(matrix.tobytes()).decode()}
            path.write_text(json.dumps(storage),encoding='utf-8')
            with self.assertRaises(ValueError): ReadOnlyNanoIndex(path)

    def test_nested_labels_rejected_but_source_words_allowed(self):
        for field in ('answer','ground_truth',' GT_Answer ','pred','score'):
            record={'evidence_id':'x','text':'pressure','metadata':[{'labels':{field:'SYNTHETIC_GOLD'}}]}
            with self.subTest(field=field),self.assertRaises(ValueError): BM25([record])
            with self.subTest(adapter='record_text',field=field),self.assertRaises(ValueError): record_text(record)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'corpus.jsonl'
                path.write_text(json.dumps({'role':'source_pdf_region',**record})+'\n',encoding='utf-8')
                with self.subTest(adapter='load_evidence',field=field),self.assertRaises(ValueError): load_evidence(path)
        text='The pressure answer and score are printed in this source document.'
        self.assertEqual(record_text({'text':text}),text)
        self.assertEqual(BM25([{'evidence_id':'x','text':text}]).search('pressure')[0]['text'],text)

    def test_bm25_output_uses_evidence_whitelist(self):
        record={'role':'source_pdf_region','evidence_id':'x','text':'pressure','page':2,
                'metadata':{'arbitrary':'not evidence'},'unknown':'not evidence',
                'aliases':[{'document_name':'example','record_index':0,'unknown':'not evidence'}]}
        hit=BM25([record]).search('pressure')[0]
        self.assertEqual(set(hit),{'route','score','role','evidence_id','text','page','aliases'})
        self.assertEqual(hit['aliases'],[{'document_name':'example','record_index':0}])
        self.assertIn('unknown',record)  # Output filtering does not mutate the input.

    def test_bm25_average_below_one_matches_formula(self):
        index=BM25([{'evidence_id':'one','text':'pressure'},{'evidence_id':'empty','text':''}])
        self.assertEqual(index.average,0.5)
        expected=math.log(2)*2.5/(1+1.5*(0.25+0.75/0.5))
        self.assertEqual(index.search('pressure')[0]['score'],round(expected,6))
        self.assertAlmostEqual(expected,0.478032538,places=8)

    def test_bm25_zero_average_has_no_results(self):
        index=BM25([{'evidence_id':'x','text':''},{'evidence_id':'y','text':'the and of'}])
        self.assertEqual(index.average,0)
        self.assertEqual(index.search('pressure'),[])

    def test_zip_preflight_validates_each_byte_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'small.zip'
            with zipfile.ZipFile(path,'w') as archive: archive.writestr('a.txt',b'1234')
            result=preflight_zip(path)
            self.assertEqual(result['entries'],1)
            self.assertEqual(result['declared_uncompressed_bytes'],4)
            self.assertEqual(result['archive_bytes'],path.stat().st_size)
            for budget in ({'max_entries':0},{'max_directory_bytes':1},
                           {'max_archive_bytes':1},{'max_uncompressed_bytes':3}):
                with self.subTest(budget=budget),self.assertRaises(ValueError): preflight_zip(path,**budget)

    def test_zip_preflight_rejects_dishonest_footer_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'dishonest.zip'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('a',b'')
                archive.writestr('b',b'')
            data=bytearray(path.read_bytes())
            footer=data.rfind(b'PK\x05\x06')
            struct.pack_into('<HH',data,footer+8,1,1)
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'actual/footer'): preflight_zip(path)
            with self.assertRaisesRegex(ValueError,'member count'): preflight_zip(path,max_entries=1)

    def test_audit_rejects_50001_members_before_hash_or_zipfile(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            large,small=root/'large.zip',root/'small.zip'
            with zipfile.ZipFile(large,'w') as archive:
                for i in range(50001): archive.writestr(f'e{i:05}',b'')
            with zipfile.ZipFile(small,'w') as archive: archive.writestr('a',b'')
            original=large.read_bytes()
            for dishonest in (False,True):
                if dishonest:
                    modified=bytearray(original)
                    footer=modified.rfind(b'PK\x05\x06')
                    struct.pack_into('<HH',modified,footer+8,1,1)
                    large.write_bytes(modified)
                for data,demo in ((large,small),(small,large)):
                    with self.subTest(dishonest=dishonest,data=data.name),\
                         patch('evidence_lab.material_audit.sha256_file',side_effect=AssertionError('No hash before preflight')),\
                         patch('evidence_lab.material_audit.zipfile.ZipFile',side_effect=AssertionError('No ZIP allocation before preflight')):
                        with self.assertRaisesRegex(ValueError,'budget'): audit_archives(data,demo,root/'output')
                        self.assertFalse((root/'output').exists())


if __name__=='__main__': unittest.main()
