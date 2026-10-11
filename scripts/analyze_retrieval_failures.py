"""Scoring-side, post-prediction failure analysis; no model calls or answer scoring."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from evidence_lab.safe_data import sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--metrics',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Refusing to overwrite failure analysis')
    predictions=json.loads(args.predictions.read_text(encoding='utf-8'))
    metrics=json.loads(args.metrics.read_text(encoding='utf-8'))
    if not predictions['complete'] or metrics['predictions_sha256']!=sha256_file(args.predictions) or metrics['labels_sha256']!=sha256_file(args.labels):
        raise ValueError('Completed scored prediction/reference binding mismatch')
    # Labels are scoring-side data only. Do not use the gold answer field.
    labels={l['sample_id']:l for l in json.loads(args.labels.read_text(encoding='utf-8'))}
    scores={c['sample_id']:c['routes'] for c in metrics['per_case']}
    routes=sorted(metrics['metrics']);groups={};failures=[];page_diversity={r:[] for r in routes}
    for case in predictions['cases']:
        sid=case['sample_id']
        types=[t for t in ('image_plus_text_as_answer','image_only','table_required','text_only') if '_'+t+'_' in sid]
        if len(types)!=1: raise ValueError('Unknown provided sample category')
        groups.setdefault(types[0],[]).append(scores[sid])
        expected={(p['document_name'],p['page']) for p in labels[sid]['reference_pages']}
        item={'sample_id':sid,'category':types[0],'reference_pages':[{'document_name':d,'page':p} for d,p in sorted(expected)],'routes':{}}
        for route in routes:
            hits=case['routes'][route][:metrics['k']]
            observed={(h['document_name'],h['page']) for h in hits};docs={d for d,p in observed}
            page_diversity[route].append(len(observed))
            score=scores[sid][route]
            item['routes'][route]={**score,'unique_returned_pages':len(observed),
                'returned_pages':[{'document_name':d,'page':p} for d,p in sorted(observed)],
                'failure_scope':None if score['all_reference_pages_hit'] else ('source_document_miss' if not docs&{d for d,p in expected} else 'reference_page_miss_or_partial'),
                'native_top5_regions':sum(bool(h.get('extraction')) for h in hits)}
        if not scores[sid]['rrf_rerank']['all_reference_pages_hit']: failures.append(item)
    grouped={}
    for category,cases in sorted(groups.items()):
        grouped[category]={'sample_count':len(cases),'metrics':{route:{
            'page_any_hit_at5':sum(s[route]['any_reference_page_hit'] for s in cases)/len(cases),
            'page_all_hit_at5':sum(s[route]['all_reference_pages_hit'] for s in cases)/len(cases),
            'macro_reference_page_coverage_at5':sum(s[route]['reference_page_coverage'] for s in cases)/len(cases)} for route in routes}}
    comparisons={}
    for left,right in [('bm25_raw','bm25'),('bm25_no_native','bm25'),('rrf_no_native','rrf'),('rrf','rrf_rerank')]:
        comparisons[left+'_to_'+right]={
            'page_any_hit_gains':[sid for sid,s in scores.items() if not s[left]['any_reference_page_hit'] and s[right]['any_reference_page_hit']],
            'page_any_hit_losses':[sid for sid,s in scores.items() if s[left]['any_reference_page_hit'] and not s[right]['any_reference_page_hit']],
            'page_all_hit_gains':[sid for sid,s in scores.items() if not s[left]['all_reference_pages_hit'] and s[right]['all_reference_pages_hit']],
            'page_all_hit_losses':[sid for sid,s in scores.items() if s[left]['all_reference_pages_hit'] and not s[right]['all_reference_pages_hit']]}
    result={'kind':'Post-scoring original-page failure analysis; no gold answer text used or model calls',
        'predictions_sha256':metrics['predictions_sha256'],'labels_sha256':metrics['labels_sha256'],
        'grouped_metrics':grouped,'comparisons':comparisons,
        'mean_unique_pages_in_first5_regions':{r:statistics.mean(v) for r,v in page_diversity.items()},
        'reranked_all_page_failure_count':len(failures),
        'reranked_failure_scopes':dict(Counter(f['routes']['rrf_rerank']['failure_scope'] for f in failures)),
        'reranked_failures':failures,
        'boundary':'Supplied30 reference-page proxy only; errors are candidates for inspection, not proof of semantic irrelevance; no answer accuracy measured'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('reranked_failures','grouped_metrics')},indent=2))


if __name__=='__main__': main()
