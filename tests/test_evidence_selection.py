import copy
import unittest
from evidence_lab.evidence_selection import select_distinct_pages

class EvidenceSelectionTests(unittest.TestCase):
    def test_same_page_duplicate_does_not_use_second_page_slot(self):
        rows=[{'evidence_id':str(i),'doc_sha256':'D','page':p,'score':10-i} for i,p in enumerate([2,2,5,8])]
        before=copy.deepcopy(rows)
        self.assertEqual([r['evidence_id'] for r in select_distinct_pages(rows)],['0','2'])
        self.assertEqual(rows,before)

    def test_falls_back_to_second_original_region_if_only_one_page(self):
        rows=[{'evidence_id':str(i),'doc_sha256':'D','page':2} for i in range(3)]
        self.assertEqual(select_distinct_pages(rows),rows[:2])
        self.assertEqual(select_distinct_pages(rows,1),rows[:1])
        self.assertEqual(select_distinct_pages([]),[])

if __name__ == '__main__':unittest.main()
