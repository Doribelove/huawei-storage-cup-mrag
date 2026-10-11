"""Read-only ZIP inventory and streaming SHA256; never extract or import content."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evidence_lab.safe_data import preflight_zip


def metadata_guard(path: Path) -> None:
    preflight_zip(path)


def issues(name: str, info: zipfile.ZipInfo) -> list[str]:
    result = []
    normalized = name.replace('\\', '/')
    parts = PurePosixPath(normalized).parts
    if normalized.startswith('/') or re.match(r'^[a-zA-Z]:', normalized) or '..' in parts:
        result.append('path_traversal_or_absolute')
    if ':' in normalized or any(ord(c) < 32 for c in normalized):
        result.append('windows_ADS_or_control_character')
    if any(part.rstrip(' .') != part for part in parts):
        result.append('windows_trailing_dot_or_space')
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{prefix}{n}' for prefix in ('COM','LPT') for n in range(1,10)}
    if any(part.split('.')[0].upper() in reserved for part in parts):
        result.append('windows_reserved_name')
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        result.append('symlink')
    if info.flag_bits & 1:
        result.append('encrypted')
    if info.file_size > 0 and info.compress_size == 0:
        result.append('nonempty_zero_compressed_size')
    if info.file_size > 1024**2 and info.file_size / max(info.compress_size,1) > 1000:
        result.append('expansion_ratio_over_1000')
    if info.file_size > 8 * 1024**3:
        result.append('single_file_over_8GiB')
    if info.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED,zipfile.ZIP_BZIP2,zipfile.ZIP_LZMA):
        result.append('unsupported_compression')
    return result


def inspect(path: Path, output: Path) -> dict:
    metadata_guard(path)
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(4 * 1024**2):
            digest.update(chunk)
    entries = []
    seen = set()
    extensions, roots = Counter(), Counter()
    interesting = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            flags = issues(info.filename, info)
            canonical = info.filename.replace('\\','/').casefold()
            if canonical in seen:
                flags.append('duplicate_casefold_path')
            seen.add(canonical)
            entry = {'path':info.filename,'bytes':info.file_size,'compressed_bytes':info.compress_size,
                     'compression':info.compress_type,'crc32':f'{info.CRC:08x}','directory':info.is_dir(),'issues':flags}
            entries.append(entry)
            if not info.is_dir():
                extensions[PurePosixPath(info.filename).suffix.lower() or '<none>'] += 1
                roots[info.filename.replace('\\','/').split('/')[0]] += 1
                if re.search(r'readme|license|requirements|说明|规则|赛题|指南|数据|index|demo|\.md$|\.ipynb$',info.filename,re.I):
                    interesting.append(entry)
    uncompressed = sum(e['bytes'] for e in entries)
    compressed = sum(e['compressed_bytes'] for e in entries)
    report = {'archive':str(path.resolve()),'archive_bytes':path.stat().st_size,'sha256':digest.hexdigest(),
              'entry_count':len(entries),'file_count':sum(not e['directory'] for e in entries),
              'declared_uncompressed_bytes':uncompressed,'compressed_member_bytes':compressed,
              'overall_expansion_ratio':round(uncompressed/max(compressed,1),3),
              'extensions':dict(extensions.most_common()),'roots':dict(roots.most_common()),
              'issue_count':sum(bool(e['issues']) for e in entries),
              'issues':[e for e in entries if e['issues']],
              'archive_warnings':['total_uncompressed_over_20GiB'] if uncompressed>20*1024**3 else [],
              'largest_files':sorted((e for e in entries if not e['directory']),key=lambda e:e['bytes'],reverse=True)[:12],
              'interesting_files':interesting,'entries':entries,
              'scope':'Metadata and archive SHA only. CRC/content integrity not yet validated; no extraction or execution.'}
    target=output/f'{path.stem}.inventory.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={k:v for k,v in report.items() if k not in ('entries','interesting_files')}
    summary['interesting_preview']=interesting[:35]
    summary['inventory_path']=str(target.resolve())
    return summary


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('archives',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for path in args.archives:
        print(json.dumps(inspect(path,args.output),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
