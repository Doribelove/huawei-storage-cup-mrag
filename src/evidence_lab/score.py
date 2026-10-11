"""Document/code-aligned score projection, not an official awarded score."""
import math


def project_score(accuracy,search_time,build_time,baseline_search,baseline_build):
    values=(accuracy,search_time,build_time,baseline_search,baseline_build)
    if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in values):
        raise ValueError('Finite numeric inputs required')
    if not 0<=accuracy<=1 or min(search_time,build_time)<0 or min(baseline_search,baseline_build)<=0:
        raise ValueError('Invalid accuracy/time/baseline range')
    search_penalty=min((search_time-baseline_search)/baseline_search,1)
    build_penalty=5*min((build_time-baseline_build)/baseline_build,1)
    return {'projected_score':100*accuracy-search_penalty-build_penalty,
            'search_penalty':search_penalty,'build_penalty':build_penalty,
            'note':'No lower clamp in supplied formula; eligibility at exactly 80/85% is unspecified.'}
