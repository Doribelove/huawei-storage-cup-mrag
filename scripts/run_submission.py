"""Question-only local submission entry; diagnostic oracle inputs are unavailable."""
import argparse
from pathlib import Path
import runpy
import sys


def main():
    project = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'model-dir', 'bge-dir', 'reranker-dir', 'questions', 'output', 'native-image-inventory'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--config', type=Path, default=project/'configs/agx-source-facts-v14.json')
    ap.add_argument('--authorization-file', type=Path)
    ap.add_argument('--max-run-seconds', type=int, default=5400)
    args = ap.parse_args()
    command = [str(project/'scripts/run_agx_batch.py')]
    for name, value in vars(args).items():
        if value is not None:
            command.extend(['--'+name.replace('_', '-'), str(value)])
    saved = sys.argv
    try:
        sys.argv = command
        runpy.run_path(command[0], run_name='__main__')
    finally:
        sys.argv = saved


if __name__ == '__main__':
    main()
