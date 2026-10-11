"""Negative-only generic metric sanity gate; never establishes positive support."""
import re

def metric_sanity(question, candidate):
    q=question.casefold()
    if not re.search(r'misclassif|classification error|错误分类|误分类',q):
        return {'passed':True,'scope':'No applicable metric substitution rule','establishes_support':False}
    answer=candidate.get('answer','')
    observations=' '.join(c.get('observation','') for c in candidate.get('citations',[]))
    text=(answer+' '+observations).casefold()
    complement=bool(re.search(r'1\s*[-−]\s*(?:[0-9.]+\s*)?f\s*[-_]?\s*1',text))
    causal=bool(re.search(r'misclassification.{0,180}(?:based on|calculated as|calculated from|derived from).{0,100}f\s*[-_]?\s*1',text))
    passed=not(complement or causal)
    return {'passed':passed,'rule':'Do not substitute1-F1 for confusion-matrix misclassification rate',
            'reason':None if passed else 'F1 complement is a different metric; a valid confusion-matrix denominator is required',
            'establishes_support':False,'question_ID_used':False,'gold_answer_used':False}
