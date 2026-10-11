"""Post-prediction paired-control analysis; no answer labels or model calls."""
import argparse
import json
from pathlib import Path
from evidence_lab.safe_data import sha256_file


def ranked(hits): return [(h['evidence_id'],h['score']) for h in hits]
def spans(note): return [w['source_token_span'] for w in note['source_windows']]
def geometry(notes):
    return [(n['evidence_id'],[(w['source_token_span'],w['source_char_span'],w['tokens_with_specials']) for w in n['source_windows']]) for n in notes]
def logits(notes): return [w['logit'] for n in notes for w in n['source_windows']]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('old-predictions','predictions','metrics','run-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Refusing to overwrite paired-control analysis')
    old=json.loads(args.old_predictions.read_text(encoding='utf-8'))
    new=json.loads(args.predictions.read_text(encoding='utf-8'))
    metrics=json.loads(args.metrics.read_text(encoding='utf-8'))
    manifest=json.loads(args.run_manifest.read_text(encoding='utf-8'));protocol=manifest['protocol']
    if not old['complete'] or not new['complete'] or not new['lineage_control_enabled'] or metrics['qa_answer_accuracy_measured']:
        raise ValueError('Require completed genuine paired-control/page-scored runs, no answer-accuracy claim')
    if sha256_file(args.old_predictions)!=protocol['old_predictions_sha256'] or sha256_file(args.predictions)!=metrics['predictions_sha256']:
        raise ValueError('Old/new prediction binding mismatch')
    for field in ('index_sha256','frozen_corpus_sha256','questions_sha256'):
        if old[field]!=new[field] or new[field]!=protocol[field]: raise ValueError('Frozen source/index/question mismatch')
    if [c['sample_id'] for c in old['cases']]!=[c['sample_id'] for c in new['cases']]: raise ValueError('Paired case order mismatch')
    scores={c['sample_id']:c['routes'] for c in metrics['per_case']};control='rrf_rerank_no_dense_lineage';fixed='rrf_rerank'
    base_routes=('bm25_raw','bm25','bm25_no_native','dense','rrf','rrf_no_native')
    base_checks={r:[] for r in base_routes};pool_checks=[];old_pool_checks=[];control_rank_checks=[];control_geometry_checks=[];control_logit_checks=[]
    changed_rankings=[];window_changes=[];positive=total=verified=omitted=control_pairs=0;per_case=[];control_logit_max_difference=0.0
    for before,after in zip(old['cases'],new['cases']):
        sid=after['sample_id'];notes=after['rerank_stats']['source_window_notes'];cn=after['lineage_control_stats']['source_window_notes'];on=before['rerank_stats']['source_window_notes']
        for route in base_routes: base_checks[route].append(ranked(before['routes'][route])==ranked(after['routes'][route]))
        pool_checks.append([n['evidence_id'] for n in notes]==[n['evidence_id'] for n in cn])
        old_pool_checks.append([n['evidence_id'] for n in on]==[n['evidence_id'] for n in cn])
        control_rank_checks.append(ranked(before['routes'][fixed])==ranked(after['routes'][control]))
        control_geometry_checks.append(geometry(on)==geometry(cn))
        control_logit_checks.append(logits(on)==logits(cn))
        if len(logits(on))!=len(logits(cn)): raise ValueError('Old/control pair count changed')
        control_logit_max_difference=max(control_logit_max_difference,max((abs(a-b) for a,b in zip(logits(on),logits(cn))),default=0))
        control_pairs+=after['lineage_control_stats']['pairs_encoded']
        changed=ranked(after['routes'][fixed])!=ranked(after['routes'][control])
        if changed: changed_rankings.append(sid)
        for fn,control_note in zip(notes,cn):
            total+=len(fn['source_windows']);positive+=sum(w['dense_overlap']>0 for w in fn['source_windows'])
            verified+=bool(fn['matched_chunk_verified_against_index_and_source_tokens']);omitted+=fn['omitted_windows']
            if spans(fn)!=spans(control_note):
                window_changes.append({'sample_id':sid,'evidence_id':fn['evidence_id'],'validated_dense_match':fn['matched_chunk'],
                    'fixed_windows':spans(fn),'control_windows':spans(control_note),
                    'selected_window_set_changed':sorted(spans(fn))!=sorted(spans(control_note)),
                    'fixed_dense_overlap':[w['dense_overlap'] for w in fn['source_windows']]})
        s=scores[sid]
        per_case.append({'sample_id':sid,'ranking_or_score_changed':changed,
            'top5_ids_changed':[h['evidence_id'] for h in after['routes'][fixed]]!=[h['evidence_id'] for h in after['routes'][control]],
            'fixed':s[fixed],'control':s[control]})
    gains_losses={}
    for metric in ('any_reference_page_hit','all_reference_pages_hit'):
        gains_losses[metric]={'gains':[c['sample_id'] for c in per_case if c['fixed'][metric] and not c['control'][metric]],
            'losses':[c['sample_id'] for c in per_case if c['control'][metric] and not c['fixed'][metric]]}
    report={'kind':'Post-scoring P2lineage fix with fresh genuine matched-pool control; previously seen development set',
        'source_commit':manifest['source_commit'],'protocol_sha256':manifest['protocol_sha256'],
        'old_predictions_sha256':sha256_file(args.old_predictions),'new_predictions_sha256':sha256_file(args.predictions),
        'sample_count':len(per_case),'base_route_ids_scores_unchanged':{r:all(v) for r,v in base_checks.items()},
        'fixed_control_candidate_id_order_identical':all(pool_checks),'old_control_candidate_id_order_identical':all(old_pool_checks),
        'old_control_top5_ids_scores_exactly_identical':all(control_rank_checks),
        'old_control_all_passage_geometry_identical':all(control_geometry_checks),
        'old_control_all_logits_exactly_identical':all(control_logit_checks),'old_control_max_absolute_logit_difference':control_logit_max_difference,
        'fresh_fixed_pairs':total,'fresh_control_pairs':control_pairs,'fixed_nonzero_dense_overlap_windows':positive,
        'validated_matched_candidates':verified,'source_windows_omitted_per_fixed_budget':omitted,
        'changed_window_orders_or_sets':len(window_changes),'changed_window_sets':sum(c['selected_window_set_changed'] for c in window_changes),
        'changed_top5_ids_cases':[c['sample_id'] for c in per_case if c['top5_ids_changed']],
        'changed_top5_rankings_or_scores_cases':changed_rankings,
        'page_metrics_fixed':metrics['metrics'][fixed],'page_metrics_control':metrics['metrics'][control],
        'gains_losses':gains_losses,'per_case':per_case,'window_changes':window_changes,
        'answer_accuracy_measured':False,'reader_run':False,
        'boundary':'Correct original-window lineage does not establish visual understanding, answer support or blind-test generalization; Agent coverage policy unchanged'}
    if not all(pool_checks) or not all(old_pool_checks) or not all(all(v) for v in base_checks.values()):
        raise ValueError('Paired-control candidate/base route invariants failed')
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_case','window_changes')},indent=2))


if __name__=='__main__': main()
