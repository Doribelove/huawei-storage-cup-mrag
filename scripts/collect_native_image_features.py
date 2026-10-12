"""Finite, resumable original-PDF image feature inventory; no QA or models."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback


def save(path,value):
    temp=path.with_suffix(path.suffix+'.new')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.project/'src'))
    import pdfplumber
    from evidence_lab.cli import load_evidence
    from evidence_lab.portable import PortableSource
    from evidence_lab.native_paint import form_bound_page
    from evidence_lab.native_image_features import image_region_features
    from evidence_lab.readable_region import valid_box
    corpus=args.data_root/'data/full/regions.jsonl'
    rows=load_evidence(corpus)
    images=sorted((r for r in rows if r['type']=='image'),
                  key=lambda r:(r['document_name'],r['page'],r['evidence_id']))
    assert len(images)==len({r['evidence_id'] for r in images})
    browser=PortableSource(args.data_root)
    frozen={'schema':'question-independent-native-image-feature-inventory-v1',
        'corpus_sha256':sha(corpus),'original_audit_sha256':sha(args.data_root/'data/audit/data-audit.json'),
        'worker_sha256':sha(Path(__file__)),
        'feature_helper_sha256':sha(args.project/'src/evidence_lab/native_image_features.py'),
        'native_filter_sha256':sha(args.project/'src/evidence_lab/native_paint.py'),
        'pdfplumber_version':importlib.metadata.version('pdfplumber'),
        'image_denominator':len(images),'image_ids':[r['evidence_id'] for r in images],
        'question_ids_or_questions_accessed':False,'reference_answers_or_pages_accessed':False,
        'model_weights_loaded':False,'original_source_records_changed':False,
        'source_pixels_or_answer_semantics_established':False}
    assert frozen['pdfplumber_version']=='0.11.9'
    args.output.mkdir(parents=True,exist_ok=True)
    freeze_path=args.output/'freeze.json'
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text())==frozen,'Resume requires identical source/code'
    else:
        assert not list(args.output.iterdir()),'New output must be empty'
        save(freeze_path,frozen)
    pages_dir=args.output/'native-pages';pages_dir.mkdir(exist_ok=True)
    cases_dir=args.output/'image-cases';cases_dir.mkdir(exist_ok=True)
    bydoc={}
    for source in images:
        bydoc.setdefault(source['document_name'],{}).setdefault(source['page'],[]).append(source)
    started=time.perf_counter();cases=[];pdfs=[]
    def summary():
        successful=[item for item in cases if item['status']=='collected']
        report={**frozen,'completed_attempts':len(cases),'failed':len(cases)-len(successful),
            'successful':len(successful),'native_labels_present':sum(item['native_region_word_count']>0 for item in successful),
            'no_native_region_text_visual_acquisition_needed':sum(item['native_region_word_count']==0 for item in successful),
            'native_words_added_to_empty_parsed_image_text':sum(item['native_region_word_count']>0 and not item['original_parsed_image_text'].strip() for item in successful),
            'verified_pdf_documents':pdfs,'elapsed_seconds':time.perf_counter()-started,
            'cases':[{k:item.get(k) for k in ('evidence_id','document_name','page','status','native_region_word_count','error')} for item in cases],
            'official_or_answer_accuracy_claimed':False,
            'limits':'Native words and captions are search metadata; quantized region association and full paint visibility are unverified; production source rows stay original.'}
        save(args.output/'summary.json',report)
        return report
    for name,selected_pages in bydoc.items():
        path=browser.verified_pdf(name)
        docsha=sha(path);assert docsha==browser.documents[name]['sha256']
        pdfs.append({'document_name':name,'pdf_sha256':docsha,'original_pdf_path':str(path)})
        with pdfplumber.open(path) as pdf:
            for number,page_sources in selected_pages.items():
                page_path=pages_dir/f'{docsha}-p{number}.json'
                page_error=None;page_data=None
                try:
                    browser.checked_page(name,number)
                    if page_path.exists():
                        page_data=json.loads(page_path.read_text())
                        assert page_data['pdf_sha256']==docsha and page_data['page']==number
                    else:
                        original=pdf.pages[number-1]
                        if original.rotation!=0 or list(original.cropbox)!=list(original.mediabox):
                            raise ValueError('Crop/rotation transform is not qualified for these features')
                        page,unfiltered,audit=form_bound_page(original)
                        words=page.extract_words()
                        if len(words)>15000:
                            raise ValueError('Native page word budget exceeded')
                        native=[];outside=[]
                        for word in words:
                            record={key:word[key] for key in ('text','x0','x1','top','bottom')}
                            (native if valid_box([word['x0'],word['top'],word['x1'],word['bottom']],original.width,original.height) else outside).append(record)
                        embedded=[{'native_object_index':i,'bbox_pdf_top_left_points':[v['x0'],v['top'],v['x1'],v['bottom']]}
                                  for i,v in enumerate(page.images)]
                        page_data={'schema':'original-native-page-search-records-v1',
                            'document_name':name,'pdf_sha256':docsha,'page':number,
                            'width':original.width,'height':original.height,'native_word_records':native,
                            'excluded_outside_page_word_records':outside,'native_paint_filter':audit,
                            'embedded_pdf_images':embedded,'source_quote_or_answer_support_established':False}
                        save(page_path,page_data)
                        original.flush_cache();unfiltered.flush_cache()
                except Exception as exc:
                    page_error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
                for source in page_sources:
                    target=cases_dir/(source['evidence_id']+'.json')
                    if target.exists():
                        item=json.loads(target.read_text())
                        assert item['evidence_id']==source['evidence_id']
                    else:
                        try:
                            if page_error is not None:
                                raise RuntimeError(json.dumps(page_error))
                            proof=browser.verify_citation(source['evidence_id'])
                            assert proof['all_passed'],'Original source citation binding failed'
                            item=image_region_features(source,page_data['native_word_records'],
                                width=page_data['width'],height=page_data['height'],
                                page_record_sha256=sha(page_path))
                            item.update(status='collected',native_page_file=str(page_path),original_source_binding=proof)
                        except Exception as exc:
                            item={'evidence_id':source['evidence_id'],'document_name':name,'page':number,
                                'status':'failed','error':{'type':type(exc).__name__,'message':str(exc)},
                                'original_parsed_image_text':source['text'],'source_quote_or_answer_fact_established':False}
                        save(target,item)
                    cases.append(item)
                report=summary()
                print(json.dumps({'document_name':name,'page':number,'completed':len(cases),
                    'denominator':len(images),'failed':report['failed'],
                    'native_labels_present':report['native_labels_present'],
                    'no_native_region_text':report['no_native_region_text_visual_acquisition_needed']}),flush=True)
        assert sha(path)==docsha,'Original PDF changed during feature collection'
    assert len(cases)==len(images)
    with (args.output/'features.jsonl').open('w') as stream:
        for item in cases:
            stream.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
    report=summary()
    report['features_sha256']=sha(args.output/'features.jsonl')
    save(args.output/'summary.json',report)
    print(json.dumps({key:report[key] for key in ('completed_attempts','image_denominator','successful','failed','native_labels_present','no_native_region_text_visual_acquisition_needed','native_words_added_to_empty_parsed_image_text','elapsed_seconds')}),flush=True)


if __name__=='__main__':
    main()
