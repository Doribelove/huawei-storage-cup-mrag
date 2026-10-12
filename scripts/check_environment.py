"""Check the explicitly supported AGX runtime without installing or downloading."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def main():
    project = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--check-cuda', action='store_true')
    args = ap.parse_args()
    expected = json.loads((project/'environment/qualified-runtime.json').read_text())
    report = {'schema': 'submission-runtime-check-v1', 'python': platform.python_version(),
              'machine': platform.machine(), 'packages': [], 'errors': [], 'cuda_checked': False,
              'fresh_independent_GPU_environment_installed': False}
    for name, version in expected['required_packages'].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        report['packages'].append({'name': name, 'expected': version, 'actual': actual})
        if actual != version:
            report['errors'].append('Unsupported or missing package: '+name)
    if sys.version_info[:2] != (3, 12):
        report['errors'].append('Python3.12 is required by the qualified runtime')
    if platform.machine() != 'aarch64':
        report['errors'].append('Only aarch64 AGX runtime has been qualified')
    if args.check_cuda:
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            with torch.inference_mode():
                actual = torch.ones((2, 2), device='cuda', dtype=torch.float16)
                assert torch.equal((actual @ actual).cpu(), torch.full((2, 2), 2, dtype=torch.float16))
            report['cuda_checked'] = True
            report['cuda_version'] = torch.version.cuda
            report['device_name'] = torch.cuda.get_device_name(0)
        except Exception as exc:
            report['errors'].append(type(exc).__name__+': '+str(exc))
    report['passed'] = not report['errors']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
