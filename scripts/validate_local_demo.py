"""Six source-derived diagnostic cases. No QA answer file is read."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlencode,urlsplit
from urllib.request import urlopen

CASES=[
    {'id':'global_text','query':'foreign direct investment','document':'','expected_document':'finance_1372325','expected_page':5,'kind':'positive_source'},
    {'id':'figure_caption','query':'SDCA Cycle','document':'education_3501298','expected_document':'education_3501298','expected_page':3,'expected_type':'image','kind':'positive_source'},
    {'id':'heading','query':'Standardization SDCA','document':'education_3501298','expected_document':'education_3501298','expected_page':3,'kind':'positive_source'},
    {'id':'no_evidence','query':'zzzxqvnonev290','document':'education_3501298','kind':'negative_source'},
    {'id':'missing_glossary_table','query':'Silent Alarm','document':'crm_5963931','expected_document':'crm_5963931','expected_page':21,'kind':'known_parse_gap'},
    {'id':'missing_responsibility_table','query':'Receiving Clerk','document':'crm_4502338','expected_document':'crm_4502338','expected_page':8,'kind':'known_parse_gap'},
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8765')
    parser.add_argument('--output',type=Path,default=Path('output/audit/phase2/retrieval-validation.json'))
    args=parser.parse_args()
    url=urlsplit(args.url)
    if url.scheme!='http' or url.hostname!='127.0.0.1': raise ValueError('Only the loopback demo is supported')
    calls=0
    def get(endpoint,parameters):
        nonlocal calls
        calls+=1
        with urlopen(args.url+endpoint+'?'+urlencode(parameters),timeout=30) as response: return json.load(response)
    results=[]
    for case in CASES:
        response=get('/api/search',{'q':case['query'],'doc':case['document'],'k':3})
        hits=response['hits']
        citations=[get('/api/verify',{'id':hit['evidence_id']}) for hit in hits]
        scope_valid=not case['document'] or all(h['document_name']==case['document'] for h in hits)
        matching=[h for h in hits if h['document_name']==case.get('expected_document') and h['page']==case.get('expected_page')
                  and ('expected_type' not in case or h['type']==case['expected_type'])]
        if case['kind']=='negative_source': expectation_met=not hits
        else: expectation_met=bool(matching)
        results.append({**case,'returned_hits':[{'evidence_id':h['evidence_id'],'document':h['document_name'],
            'page':h['page'],'type':h['type'],'bm25_score':h['score']} for h in hits],
            'source_expectation_met':expectation_met,'document_scope_valid':scope_valid,
            'latency_ms':response['latency_ms'],'citations':citations,
            'all_returned_citations_passed':all(c['all_passed'] for c in citations),
            'answer_generated':response['answer_generated'],
            'interpretation':'Expected known parser coverage gap; whole-page fallback remains viewable' if case['kind']=='known_parse_gap' and not expectation_met else 'Source diagnostic only'})
    report={'kind':'six manually selected source diagnostics; not an independent benchmark or competition Accuracy',
            'case_selection':'Queries and page expectations taken from reviewed source text/figures; never queries_30.json gold',
            'cases':results,'all_citation_checks_passed':all(r['all_returned_citations_passed'] for r in results),
            'all_document_scope_checks_passed':all(r['document_scope_valid'] for r in results),
            'basic_positive_and_negative_cases_passed':all(r['source_expectation_met'] for r in results if r['kind']!='known_parse_gap'),
            'observed_parse_coverage_gaps':[r['id'] for r in results if r['kind']=='known_parse_gap' and not r['source_expectation_met']],
            'localhost_http_requests':calls,'external_network_requests':0,'model_calls':0,'gold_answers_read':False,
            'limitations':['Source-selected cases cannot estimate generalization or QA accuracy','Document-scoped BM25 uses that document for IDF/length normalization',
                          'Missing table bodies remain unresolved; fallback images are not converted to guessed searchable text']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key!='cases'},ensure_ascii=False,indent=2))
    if not all(report[key] for key in ('all_citation_checks_passed','all_document_scope_checks_passed','basic_positive_and_negative_cases_passed')):
        raise SystemExit(1)


if __name__=='__main__': main()
