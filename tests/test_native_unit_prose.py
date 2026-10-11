"""Literal actual-PDF unit and continuation diagnostics, no benchmark claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import pdfplumber
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from evidence_lab.native_unit_prose import requested_units,native_unit_prose_plan,continuation_source,native_lines
from evidence_lab.native_pdf import SourceVerifier
from evidence_lab.reader_context import native_window_projection
from evidence_lab.safe_data import sha256_file
from evidence_lab.visual_evidence import VisualEvidence


QUESTION='What are the nanometer-scale layer dimensions for the rear treatment?'


def printed_fixture(root,finished=False,bridge=True):
    path=Path(root)/'own-measurement.pdf';c=canvas.Canvas(str(path),pagesize=(600,800))
    c.setFont('Helvetica',9)
    for y,text in ((125,'3.2 Measurement layers'),(105,'Treatment rear thin layers are deposited.'),
                   (93,'Inner layer thickness is 12 nm with fixed coating.'),
                   (81,'Middle layer thickness is 18 nm at the rear'),
                   (69,'boundary layer.' if finished else 'boundary layer')):c.drawString(320,y,text)
    c.drawString(298,20,'1');c.showPage();c.setFont('Helvetica',9)
    following='Outer layer thickness is 24 nm. Rear '+('boundary ' if bridge else '')+'treatment layers continue.'
    c.drawString(40,730,following)
    c.drawString(298,20,'2');c.save()
    doc={'sha256':sha256_file(path),'member':'own/measurement.pdf','page_count':2}
    rows=[{'evidence_id':'own-'+str(page),'document_name':'own','doc_sha256':doc['sha256'],
        'page':page,'page_idx':page-1,'type':'text','text':text,
        'bbox_points_estimate':[320,660,590,740] if page==1 else [38,60,580,76]}
        for page,text in ((1,'Entire original local measurement source region.'),
            (2,following))]
    class Browser:
        audit={'source_archives':{'data':{'sha256':'diagnostic-only'}}}
        documents={'own':doc};byid={r['evidence_id']:r for r in rows};records=rows
        def checked_page(self,name,page):
            if name!='own' or page not in (1,2):raise ValueError('Unknown diagnostic source')
            return doc
        def verified_pdf(self,name):
            if name!='own':raise ValueError('Unknown diagnostic source')
            return path
        def verify_citation(self,eid):return {'all_passed':eid in self.byid,'diagnostic_only':True}
    return path,rows,Browser()


class NativeUnitProseTests(unittest.TestCase):
    def test_explicit_unit_cues_do_not_use_answers_or_create_values(self):
        self.assertEqual(requested_units('What nanometer-scale and micrometer dimensions are used?'),['nm','um'])
        self.assertEqual(requested_units('At what voltage in volts?'),['V'])
        self.assertEqual(requested_units('Which flow steps occur?'),[])

    def test_real_native_lines_keep_literal_values_title_column_and_unfinished_tail(self):
        with tempfile.TemporaryDirectory() as root:
            path,_,_=printed_fixture(root)
            with pdfplumber.open(path) as pdf:words=pdf.pages[0].extract_words()
            before=copy.deepcopy(words);plan=native_unit_prose_plan(QUESTION,words,600,800)
            self.assertEqual(words,before)
            self.assertTrue(plan['next_page_context_hint'])
            text=' '.join(line['text'] for line in plan['lines'])
            for literal in ('3.2 Measurement layers','12 nm','18 nm','boundary layer'):
                self.assertIn(literal,text)
            for line in plan['lines']:
                self.assertEqual(line['text'],' '.join(words[i]['text'] for i in line['native_word_indices']))
            self.assertNotEqual(plan['last_actual_body_line']['text'],'1')
            self.assertFalse(plan['answer_fact_or_arithmetic_generated'])
            self.assertFalse(plan['cross_page_continuation_verified'])

    def test_finished_source_or_no_explicit_unit_does_not_request_next_page(self):
        with tempfile.TemporaryDirectory() as root:
            path,_,_=printed_fixture(root,finished=True)
            with pdfplumber.open(path) as pdf:words=pdf.pages[0].extract_words()
            self.assertFalse(native_unit_prose_plan(QUESTION,words,600,800)['next_page_context_hint'])
            self.assertIsNone(native_unit_prose_plan('What treatment occurs?',words,600,800))
            bad=copy.deepcopy(words);bad[0]['x0']=True
            self.assertIsNone(native_unit_prose_plan(QUESTION,bad,600,800))

    def test_continuation_uses_only_next_verified_document_page_and_literal_source_bridge(self):
        with tempfile.TemporaryDirectory() as root:
            path,rows,browser=printed_fixture(root)
            with pdfplumber.open(path) as pdf:words=pdf.pages[0].extract_words()
            plan=native_unit_prose_plan(QUESTION,words,600,800)
            next_source=continuation_source(QUESTION,plan,rows[0],browser,{(rows[0]['doc_sha256'],1)})
            self.assertEqual(next_source['canonical_source'],rows[1])
            self.assertFalse(next_source['acquisition_reason']['cross_page_continuation_verified'])
            self.assertIsNone(continuation_source(QUESTION,plan,rows[0],browser,{(rows[0]['doc_sha256'],2)}))
            self.assertIsNone(continuation_source(QUESTION,plan,rows[1],browser,set()))

    def test_actual_render_persist_verify_and_reader_projection_keep_whole_original_region(self):
        with tempfile.TemporaryDirectory() as root:
            _,rows,browser=printed_fixture(root);visual=VisualEvidence(SourceVerifier(browser))
            old=visual.prepare_region(rows[0]['evidence_id'],Path(root)/'legacy',
                readable_question=QUESTION,full_page_recovery=True)
            target=Path(root)/'new'
            new=visual.prepare_region(rows[0]['evidence_id'],target,readable_question=QUESTION,
                full_page_recovery=True,native_unit_prose_recovery=True)
            self.assertEqual(old['page_image'],new['page_image'])
            self.assertEqual(old['geometry'],new['geometry'])
            self.assertTrue(visual.verify(target)['all_passed'])
            self.assertEqual(new,json.loads((target/'binding.json').read_text()))
            projected=native_window_projection(new['readable_viewport'],rows[0]['text'])
            self.assertEqual(projected['original_source_region_transcription'],rows[0]['text'])
            self.assertIn('12 nm',' '.join(projected['NATIVE_UNIT_PROSE_LINES']))
            self.assertFalse(new['supported_answer'])
            self.assertEqual(native_window_projection(old['readable_viewport'],rows[0]['text']),rows[0]['text'])

    def test_native_two_columns_and_empty_paragraph_repair_do_not_merge_neighbors(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'own-columns.pdf'
            c=canvas.Canvas(str(path),pagesize=(600,800));c.setFont('Helvetica',9)
            c.drawString(40,730,'Layer is 24 nm.')
            c.drawString(40+stringWidth('Layer is 24 nm.','Helvetica',9)+18,730,'Other treatment layer is 99 nm.')
            c.save()
            with pdfplumber.open(path) as pdf:words=pdf.pages[0].extract_words()
            lines=native_lines(words,600,800)
            self.assertEqual([line['text'] for line in lines],['Layer is 24 nm.','Other treatment layer is 99 nm.'])
            plan=native_unit_prose_plan(QUESTION,words,600,800,
                source_box=[38,60,108,76],source_transcription='')
            self.assertTrue(plan['empty_source_transcription_repaired_from_native_rectangle'])
            self.assertEqual([line['text'] for line in plan['lines']],['Layer is 24 nm.'])
            self.assertFalse(plan['next_page_context_hint'])
            self.assertFalse(plan['source_bbox_association_semantics_verified'])
            self.assertIsNone(native_unit_prose_plan(QUESTION,words,600,800,
                source_box=[300,60,400,76],source_transcription=''))

    def test_continuation_uses_real_words_when_parsed_transcription_is_empty(self):
        with tempfile.TemporaryDirectory() as root:
            path,rows,browser=printed_fixture(root)
            rows[1]['text']=''
            with pdfplumber.open(path) as pdf:words=pdf.pages[0].extract_words()
            plan=native_unit_prose_plan(QUESTION,words,600,800)
            found=continuation_source(QUESTION,plan,rows[0],browser,set())
            self.assertEqual(found['canonical_source'],rows[1])
            self.assertIn('24 nm',' '.join(line['text'] for line in found['acquisition_reason']['actual_next_source_native_lines']))
            self.assertFalse(found['acquisition_reason']['cross_page_continuation_verified'])
            visual=VisualEvidence(SourceVerifier(browser));target=Path(root)/'repair'
            binding=visual.prepare_region('own-2',target,readable_question=QUESTION,
                full_page_recovery=True,native_unit_prose_recovery=True)
            self.assertTrue(visual.verify(target)['all_passed'])
            projection=native_window_projection(binding['readable_viewport'],'')
            self.assertEqual(projection['original_source_region_transcription'],'')
            self.assertIn('24 nm',' '.join(projection['NATIVE_UNIT_PROSE_LINES']))

    def test_single_generic_bridge_word_is_insufficient_for_continuation(self):
        with tempfile.TemporaryDirectory() as root:
            path,rows,browser=printed_fixture(root,bridge=False)
            with pdfplumber.open(path) as pdf:words=pdf.pages[0].extract_words()
            plan=native_unit_prose_plan(QUESTION,words,600,800)
            self.assertIsNone(continuation_source(QUESTION,plan,rows[0],browser,set()))


if __name__=='__main__':unittest.main()
