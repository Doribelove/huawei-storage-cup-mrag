import argparse
from collections import defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
from pypdf import PdfReader
from evidence_lab.safe_data import preflight_zip,read_member,member_json,record_text

parser=argparse.ArgumentParser(description='Render known source anomaly pages; preserve originals')
parser.add_argument('--audit-dir',type=Path,default=Path('output/audit'))
parser.add_argument('--reviews',type=Path,default=Path('docs/phase2-page-reviews.json'))
args=parser.parse_args()
root=args.audit_dir
out=root/'phase2'
pages=out/'pages'
pages.mkdir(parents=True,exist_ok=True)
audit=json.loads((root/'data-audit.json').read_text(encoding='utf-8'))
quality=json.loads((root/'data-quality-details.json').read_text(encoding='utf-8'))
targets=defaultdict(list)
for doc,nums in quality['PDF_pages_without_parsed_records'].items():
    for p in nums: targets[(doc,p)].append({'kind':'no_parsed_records'})
for entry in quality['empty_image_refs']:
    targets[(entry['document'],entry['page'])].append({'kind':'empty_table','record_index':entry['record_index']})
for entry in quality['invalid_region_details']:
    targets[(entry['document'],entry['page'])].append({'kind':'invalid_bbox','record_index':entry['record_index'],'source_bbox':entry['bbox']})
data=Path(audit['source_archives']['data']['path'])
preflight_zip(data)
results=[]
with zipfile.ZipFile(data) as archive:
    for (name,num),issues in sorted(targets.items()):
        doc=audit['dataset']['documents'][name]
        raw=read_member(archive,doc['member'],32*1024**2)
        assert hashlib.sha256(raw).hexdigest()==doc['sha256']
        source=pages/(name+'.pdf')
        if source.exists() and source.read_bytes()!=raw: raise ValueError('Cached source PDF changed; refusing overwrite')
        if not source.exists(): source.write_bytes(raw)
        records=member_json(archive,doc['content_list']['member'])
        page=PdfReader(BytesIO(raw)).pages[num-1]
        text=page.extract_text() or ''
        prefix=pages/f'{name}-p{num}'
        subprocess.run([shutil.which('pdftoppm'),'-f',str(num),'-l',str(num),'-singlefile','-r','90','-png',str(source),str(prefix)],
                       check=True,capture_output=True,timeout=30)
        for issue in issues:
            if 'record_index' in issue:
                record=records[issue['record_index']]
                issue.update({'source_bbox':record['bbox'],'source_type':record['type'],
                              'source_caption':record_text(record),'image_path':record.get('img_path')})
        png=prefix.with_suffix('.png')
        result={'document_name':name,'page':num,'doc_sha256':doc['sha256'],
                'pdf_member':doc['member'],'page_size_points':[float(page.mediabox.width),float(page.mediabox.height)],
                'page_rotation':int(page.get('/Rotate',0)),'cropbox':[float(x) for x in page.cropbox],
                'native_text_characters':len(text.strip()),'parsed_region_count':sum(r['page_idx']==num-1 for r in records),
                'pdf_image_count':len(page.images),'render_path':str(png.resolve()),
                'render_sha256':hashlib.sha256(png.read_bytes()).hexdigest(),'render_dpi':90,
                'anomalies':issues,'visual_classification':'uncertain','review_note':'pending visual inspection'}
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('render_sha256','doc_sha256','pdf_member')},ensure_ascii=False),flush=True)
(out/'page-review-pending.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if args.reviews.exists():
    recorded=json.loads(args.reviews.read_text(encoding='utf-8'))
    bypage={(r['document_name'],r['page']):r for r in recorded}
    for row in results:
        old=bypage.get((row['document_name'],row['page']))
        if not old or any(row[key]!=old[key] for key in ('doc_sha256','render_sha256')):
            raise ValueError('Recorded visual review does not match source/render bytes; inspect new pages before classifying')
        row.update(old)
    (out/'page-review.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
