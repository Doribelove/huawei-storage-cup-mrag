"""Export all final answers/refusals from a complete batch, never bound candidates."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from evidence_lab.evaluation import load_questions
from evidence_lab.submission_io import export_final_answers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('questions', 'batch', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    output = export_final_answers(load_questions(args.questions), args.batch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'question_count': output['question_count'],
                      'official_schema_confirmed': False}))


if __name__ == '__main__':
    main()
