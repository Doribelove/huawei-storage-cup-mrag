import tempfile,unittest
from pathlib import Path
from pypdf import PdfWriter
from pypdf.generic import NameObject,DictionaryObject,DecodedStreamObject,ArrayObject,FloatObject
import pdfplumber
from evidence_lab.native_paint import form_bound_page,disjoint

def stream(text):
    value=DecodedStreamObject();value.set_data(text.encode('ascii'));return value

def fixture(path,nested=False):
    writer=PdfWriter();page=writer.add_blank_page(width=300,height=300)
    font=writer._add_object(DictionaryObject({NameObject('/Type'):NameObject('/Font'),
        NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')}))
    resources=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):font})})
    inner=stream('BT /F1 10 Tf 1 0 0 1 40 80 Tm (INSIDE) Tj 1 0 0 1 40 160 Tm (PHANTOM) Tj ET')
    inner.update({NameObject('/Type'):NameObject('/XObject'),NameObject('/Subtype'):NameObject('/Form'),
        NameObject('/BBox'):ArrayObject([FloatObject(x) for x in [20,50,140,120]]),NameObject('/Resources'):resources})
    ref=writer._add_object(inner)
    if nested:
        outer=stream('/Inner Do');outer.update({NameObject('/Type'):NameObject('/XObject'),NameObject('/Subtype'):NameObject('/Form'),
            NameObject('/BBox'):ArrayObject([FloatObject(x) for x in [20,100,140,200]]),
            NameObject('/Resources'):DictionaryObject({NameObject('/XObject'):DictionaryObject({NameObject('/Inner'):ref})})})
        ref=writer._add_object(outer)
    page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):resources['/Font'],
        NameObject('/XObject'):DictionaryObject({NameObject('/Form'):ref})})
    page[NameObject('/Contents')]=writer._add_object(stream('q 1 0 0 1 40 10 cm /Form Do Q BT /F1 10 Tf 3 Tr 1 0 0 1 40 250 Tm (HIDDEN) Tj 0 Tr 1 0 0 1 40 220 Tm (VISIBLE) Tj ET'))
    writer.write(path)

class NativePaintTests(unittest.TestCase):
    def test_actual_form_bbox_transform_and_mode_with_raw_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'source.pdf';fixture(path)
            with pdfplumber.open(path) as pdf:
                filtered,raw,audit=form_bound_page(pdf.pages[0])
                text=' '.join(w['text'] for w in filtered.extract_words())
                self.assertIn('INSIDE',text);self.assertIn('VISIBLE',text)
                self.assertNotIn('PHANTOM',text);self.assertNotIn('HIDDEN',text)
                rawtext=' '.join(w['text'] for w in raw.extract_words())
                self.assertIn('PHANTOM',rawtext);self.assertIn('HIDDEN',rawtext)
                self.assertEqual(audit['excluded_chars'],len('PHANTOMHIDDEN'))
                inside=next(c for c in raw.chars if c['text']=='I')
                self.assertEqual(inside['source_form_constraints'][0]['bbox_pdf_layout'],[60,60,180,130])

    def test_nested_form_intersection_excludes_outer_clipped_inner_text(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'source.pdf';fixture(path,nested=True)
            with pdfplumber.open(path) as pdf:
                filtered,raw,audit=form_bound_page(pdf.pages[0])
                text=' '.join(w['text'] for w in filtered.extract_words())
                self.assertEqual(text,'VISIBLE')
                self.assertEqual(audit['excluded_chars'],len('PHANTOMHIDDENINSIDE'))

    def test_touching_or_partial_bounds_do_not_establish_visibility(self):
        self.assertFalse(disjoint([0,0,5,5],[5,0,10,5]))
        self.assertFalse(disjoint([0,0,6,5],[5,0,10,5]))
        self.assertFalse(disjoint([3,5,7,5],[0,0,10,10]))
        self.assertFalse(disjoint([5,3,5,7],[0,0,10,10]))
        self.assertTrue(disjoint([3,12,7,12],[0,0,10,10]))

