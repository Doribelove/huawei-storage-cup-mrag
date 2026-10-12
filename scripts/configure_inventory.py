"""Bind an unchanged QA configuration to an explicitly rebuilt source inventory."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from evidence_lab.submission_io import configure_inventory


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('template', 'data-root', 'inventory', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    config = configure_inventory(args.template, args.data_root, args.inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'changed_fields': ['native_image_features_sha256'],
                      'model_or_answer_rules_changed': False}))


if __name__ == '__main__':
    main()
