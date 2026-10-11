"""Source-geometry noise policy. Evidence remains available; no answerability threshold."""
from __future__ import annotations
from collections import defaultdict
import re
from .safe_data import evidence_output


class CandidatePolicy:
    def __init__(self, records: list[dict]):
        rows = [evidence_output(r) for r in records]
        self.byid = {r['evidence_id']: r for r in rows}
        if len(self.byid) != len(rows):
            raise ValueError('Duplicate source evidence ID')
        repeated = defaultdict(set)
        for row in rows:
            text = row['text'].strip().casefold()
            if row['type'] == 'text' and self.at_page_edge(row) and len(text) <= 100:
                repeated[(row['doc_sha256'], row['document_name'], text)].add(row['page'])
        self.excluded = {}
        for row in rows:
            text = row['text'].strip()
            if not text:
                self.excluded[row['evidence_id']] = 'empty_source_text'
            elif row['type'] == 'text' and self.at_page_edge(row):
                if re.fullmatch(r'(?:page\s*)?\d+(?:\s*(?:/|of)\s*\d+)?', text, re.IGNORECASE):
                    self.excluded[row['evidence_id']] = 'page_edge_number'
                elif len(text) <= 100 and len(repeated[(row['doc_sha256'], row['document_name'], text.casefold())]) >= 3:
                    self.excluded[row['evidence_id']] = 'repeated_page_edge_text'

    @staticmethod
    def at_page_edge(row: dict) -> bool:
        bbox = row.get('bbox_normalized')
        return bool(bbox and len(bbox) == 4 and (bbox[3] <= .08 or bbox[1] >= .90))

    def accepts(self, row: dict) -> bool:
        eid = row['evidence_id']
        if eid not in self.byid:
            raise ValueError('Unknown source candidate')
        return eid not in self.excluded

    def searchable(self) -> list[dict]:
        return [row for eid, row in self.byid.items() if eid not in self.excluded]

    def report(self) -> dict:
        from collections import Counter
        return {'source_regions': len(self.byid), 'searchable_regions': len(self.byid)-len(self.excluded),
                'excluded_by_reason': dict(Counter(self.excluded.values())),
                'policy': 'Empty fields and geometric page-edge noise; source records retained',
                'answerability_threshold': None, 'threshold_calibrated': False}
