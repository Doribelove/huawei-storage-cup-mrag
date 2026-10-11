from __future__ import annotations

import hashlib
import json
import re
import stat
import struct
from pathlib import Path, PurePosixPath
import zipfile

FORBIDDEN_EVIDENCE_FIELDS=frozenset({'answer','ground_truth','gt_answer','pred','score'})
EVIDENCE_OUTPUT_FIELDS=frozenset({'role','evidence_id','doc_sha256','document_name','page','page_idx','type',
    'source_bbox','bbox_normalized','bbox_points_estimate','bbox_coordinate_system','page_size_points',
    'page_rotation','text','image_member','source_archive_sha256','aliases','limits','extraction'})
ALIAS_OUTPUT_FIELDS=frozenset({'document_name','pdf_member','content_list_member','record_index','image_member'})


def reject_answer_fields(record:dict)->None:
    """Structured labels are forbidden at any depth; ordinary source text is untouched."""
    if not isinstance(record,dict): raise ValueError('Evidence record must be an object')
    pending=[record]
    seen=set()
    while pending:
        value=pending.pop()
        if not isinstance(value,(dict,list,tuple)): continue
        if id(value) in seen: continue
        seen.add(id(value))
        if isinstance(value,dict):
            if any(isinstance(key,str) and key.strip().casefold() in FORBIDDEN_EVIDENCE_FIELDS for key in value):
                raise ValueError('QA/answer fields cannot enter evidence adapter')
            pending.extend(value.values())
        else: pending.extend(value)


def evidence_output(record:dict)->dict:
    reject_answer_fields(record)
    selected={key:value for key,value in record.items() if key in EVIDENCE_OUTPUT_FIELDS}
    if 'aliases' in selected:
        selected['aliases']=[{key:value for key,value in alias.items() if key in ALIAS_OUTPUT_FIELDS}
                             for alias in selected['aliases'] if isinstance(alias,dict)]
    return selected


def preflight_zip(path:Path,*,max_entries:int=50000,max_directory_bytes:int=128*1024**2,
                  max_archive_bytes:int=4*1024**3,max_uncompressed_bytes:int=20*1024**3)->dict:
    """Bound the directory before zipfile allocates it; verify actual central record count."""
    archive_bytes=path.stat().st_size
    if archive_bytes>max_archive_bytes: raise ValueError('Archive exceeds byte budget')
    with path.open('rb') as source:
        tail_start=max(0,archive_bytes-65557)
        source.seek(tail_start)
        tail=source.read(65557)
        position=tail.rfind(b'PK\x05\x06')
        if position<0 or position+22>len(tail): raise ValueError('Missing/truncated ZIP directory footer')
        _,disk,directory_disk,disk_count,count,size,offset,comment=struct.unpack_from('<4s4H2LH',tail,position)
        if position+22+comment!=len(tail): raise ValueError('ZIP footer/comment length mismatch')
        if disk or directory_disk or disk_count!=count: raise ValueError('Multi-disk ZIP not supported')
        if count==65535 or size==0xffffffff or offset==0xffffffff:
            raise ValueError('ZIP64 requires separately bounded review')
        if count>max_entries or size>max_directory_bytes: raise ValueError('ZIP directory exceeds member/metadata budget')
        if offset+size!=tail_start+position: raise ValueError('ZIP directory offset/size mismatch')
        source.seek(offset)
        consumed=actual_count=uncompressed=0
        while consumed<size:
            header=source.read(46)
            if len(header)!=46: raise ValueError('Truncated central directory record')
            fields=struct.unpack('<4s6H3I5H2I',header)
            if fields[0]!=b'PK\x01\x02': raise ValueError('Invalid central directory record')
            variable_bytes=sum(fields[10:13])
            if consumed+46+variable_bytes>size: raise ValueError('Central record exceeds directory budget')
            actual_count+=1
            uncompressed+=fields[9]
            if actual_count>max_entries or uncompressed>max_uncompressed_bytes:
                raise ValueError('ZIP member count or uncompressed-byte budget exceeded')
            source.seek(variable_bytes,1)
            consumed+=46+variable_bytes
        if consumed!=size or actual_count!=count: raise ValueError('ZIP actual/footer member count mismatch')
    return {'entries':actual_count,'directory_bytes':size,'archive_bytes':archive_bytes,
            'declared_uncompressed_bytes':uncompressed}


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as source:
        while chunk:=source.read(4*1024**2): digest.update(chunk)
    return digest.hexdigest()


def validate_name(name: str) -> str:
    normalized=name.replace('\\','/')
    parts=PurePosixPath(normalized).parts
    reserved={'CON','PRN','AUX','NUL'}|{f'{p}{i}' for p in ('COM','LPT') for i in range(1,10)}
    if not parts or normalized.startswith('/') or '..' in parts or ':' in normalized:
        raise ValueError('Absolute, drive, ADS or traversal member path')
    if any(ord(c)<32 for c in normalized): raise ValueError('Control character in path')
    if any(p.rstrip(' .')!=p or p.split('.')[0].upper() in reserved for p in parts):
        raise ValueError('Unsafe Windows path component')
    return normalized


def read_member(archive: zipfile.ZipFile, name: str, limit: int) -> bytes:
    validate_name(name)
    info=archive.getinfo(name)
    if stat.S_ISLNK(info.external_attr>>16) or info.flag_bits&1:
        raise ValueError('Symlink/encrypted member')
    if info.file_size>limit or (info.file_size>1024**2 and info.file_size/max(info.compress_size,1)>1000):
        raise ValueError('Member exceeds size/expansion budget')
    with archive.open(info) as source: data=source.read(limit+1)
    if len(data)!=info.file_size or len(data)>limit: raise ValueError('Length budget mismatch')
    return data  # zipfile checks member CRC when read to EOF.


def member_json(archive: zipfile.ZipFile, name: str, limit: int=2*1024**2):
    return json.loads(read_member(archive,name,limit).decode('utf-8-sig'))


def document_name(path: str) -> str:
    stem=PurePosixPath(path.replace('\\','/')).stem
    for suffix in ('_content_list','_origin'):
        if stem.endswith(suffix): stem=stem[:-len(suffix)]
    return stem


def record_text(record: dict) -> str:
    reject_answer_fields(record)
    parts=[]
    for key in ('text','table_body','image_caption','image_footnote','table_caption','table_footnote'):
        value=record.get(key,'')
        if isinstance(value,list): parts.extend(str(v) for v in value)
        elif isinstance(value,str): parts.append(value)
    return '\n'.join(p for p in parts if p)


def normalized_bbox(record: dict) -> list[float]:
    bbox=record.get('bbox')
    if not isinstance(bbox,list) or len(bbox)!=4 or any(isinstance(x,bool) or not isinstance(x,(int,float)) for x in bbox):
        raise ValueError('Malformed MinerU bbox')
    import math
    if not all(math.isfinite(x) for x in bbox) or not(0<=bbox[0]<=bbox[2]<=1000 and 0<=bbox[1]<=bbox[3]<=1000):
        raise ValueError('MinerU bbox outside [0,1000] or inverted')
    return [x/1000 for x in bbox]


def image_member(record: dict, content_member: str, members: set[str]) -> str|None:
    value=record.get('img_path')
    if not value: return None
    # Original paths may refer to the author's computer. Remap only within this document's images folder.
    basename=PurePosixPath(str(value).replace('\\','/')).name
    validate_name(basename)
    if Path(basename).suffix.lower() not in ('.jpg','.jpeg','.png','.webp'): raise ValueError('Unexpected image type')
    candidate=str(PurePosixPath(content_member).parent/'images'/basename)
    return candidate if candidate in members else None
