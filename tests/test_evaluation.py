"""Metric/isolation fixtures; not an accuracy benchmark."""
import json
from pathlib import Path
import tempfile
import unittest
from evidence_lab.evaluation import load_questions, reference_pages, page_reference_metrics


class EvaluationTests(unittest.TestCase):
    def test_runtime_projection_rejects_gold_references_and_duplicate_ids(self):
        good={'version':'question-only-v1','questions':[{'sample_id':'one','question':'Question?'}]}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'questions.json'
            path.write_text(json.dumps(good),encoding='utf-8')
            self.assertEqual(len(load_questions(path)),1)
            for payload in [
                {**good,'answer':'gold'},
                {**good,'questions':[{'sample_id':'one','question':'Question?','reference_pages':[1]}]},
                {**good,'questions':good['questions']*2},
                {**good,'questions':[{'sample_id':'one','question':'x'*501}]}]:
                path.write_text(json.dumps(payload),encoding='utf-8')
                with self.assertRaises(ValueError): load_questions(path)

    def test_reference_page_alignment_requires_complete_one_based_document(self):
        documents={'finance_123':{'page_count':3}}
        item={'domain':'finance','gt_image_paths':['/never/open/123_page_0002.png'],
              'longdoc_image_paths':[f'/never/open/123_page_{page:04}.png' for page in (1,2,3)]}
        pages,alignment=reference_pages(item,documents)
        self.assertEqual(pages,[{'document_name':'finance_123','page':2}])
        self.assertTrue(alignment['one_based_full_document_page_alignment'])
        for bad in [{**item,'gt_image_paths':['123_page_0000.png']},
                    {**item,'longdoc_image_paths':item['longdoc_image_paths'][:2]},
                    {**item,'gt_image_paths':['unsupported.png']}]:
            with self.assertRaises(ValueError): reference_pages(bad,documents)

    def test_page_proxy_counts_unique_pages_and_has_explicit_denominator(self):
        predictions=[{'sample_id':'one','routes':{'dense':[{'document_name':'d','page':1},{'document_name':'d','page':1}]}},
                     {'sample_id':'two','routes':{'dense':[{'document_name':'d','page':3}]}}]
        labels=[{'sample_id':'one','reference_pages':[{'document_name':'d','page':1},{'document_name':'d','page':2}]},
                {'sample_id':'two','reference_pages':[{'document_name':'d','page':3}]}]
        result=page_reference_metrics(predictions,labels,2)
        self.assertEqual(result['metrics']['dense'],{'page_any_hit_at_k':1,'page_all_hit_at_k':.5,'macro_reference_page_coverage_at_k':.75})
        self.assertEqual(result['sample_count'],2)
        self.assertIn('not labeled region Recall',result['limitations'][0])
        with self.assertRaises(ValueError): page_reference_metrics(predictions[:1],labels)
        with self.assertRaises(ValueError): page_reference_metrics(predictions*2,labels)


if __name__=='__main__': unittest.main()
