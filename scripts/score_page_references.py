"""Independent post-retrieval scorer. QA labels are never a retriever input."""
import argparse
import json
from pathlib import Path
from evidence_lab.evaluation import page_reference_metrics
from evidence_lab.safe_data import sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--evaluation-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Refusing to overwrite a scored result')
    manifest=json.loads(args.evaluation_manifest.read_text(encoding='utf-8'))
    predictions=json.loads(args.predictions.read_text(encoding='utf-8'))
    if not predictions['complete'] or predictions['questions_sha256']!=manifest['questions_sha256']:
        raise ValueError('Question/prediction binding mismatch')
    if sha256_file(args.labels)!=manifest['scorer_only_labels_sha256']:
        raise ValueError('Frozen label binding mismatch')
    # This is the only scorer label read. Gold answers are not consumed by page metrics.
    labels=json.loads(args.labels.read_text(encoding='utf-8'))
    result=page_reference_metrics(predictions['cases'],labels,5)
    result.update(predictions_sha256=sha256_file(args.predictions),labels_sha256=sha256_file(args.labels),
                  qa_answer_accuracy_measured=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='per_case'},indent=2))


if __name__=='__main__': main()
