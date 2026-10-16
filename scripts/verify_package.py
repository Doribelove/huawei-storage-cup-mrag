"""Verify all declared release files; --strict also rejects unlisted files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(root, strict=False):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest/files.json').read_text(encoding='utf-8'))
    declared = set()
    errors = []
    for entry in manifest['files']:
        name = entry['path']
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or name in declared or '\\' in name:
            raise ValueError('Invalid or duplicate manifest path')
        declared.add(name)
        local = root/name
        if local.is_symlink() or not local.resolve().is_relative_to(root) or not local.is_file():
            errors.append(name + ': missing or unsafe')
            continue
        digest = hashlib.sha256()
        with local.open('rb') as stream:
            while block := stream.read(4*1024**2): digest.update(block)
        if local.stat().st_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
            errors.append(name + ': bytes changed')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    extras = sorted(actual - declared - {'manifest/files.json','manifest/files.sha256'})
    if strict and extras:
        errors.extend(name + ': unlisted' for name in extras)
    return {'passed': not errors, 'files_checked': len(declared), 'errors': errors, 'unlisted_count': len(extras)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--strict', action='store_true')
    args = ap.parse_args()
    result = verify(Path(__file__).resolve().parents[1], args.strict)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)
