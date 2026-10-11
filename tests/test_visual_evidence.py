"""Actual local PDF raster/provenance diagnostics; no vision/model benchmark."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import FloatObject, NameObject, RectangleObject
from reportlab.pdfgen import canvas

from evidence_lab.native_pdf import SourceVerifier, native_tables
from evidence_lab.safe_data import sha256_file
from evidence_lab.visual_evidence import VisualEvidence, rectangle, transform


def fixture(root):
    root = Path(root)
    original = root/'original.pdf'
    c = canvas.Canvas(str(original), pagesize=(612, 792))
    for index in range(6):
        c.setFillColorRGB(1, 0, 0)
        c.rect(120, 140, 80, 70, fill=1, stroke=0)
        c.setFillColorRGB(0, 1, 0)
        c.rect(300, 300, 60, 60, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(130, 250, 'OWN DIAGNOSTIC PAGE '+str(index+1))
        for x in (50, 250, 550): c.line(x, 600, x, 700)
        for y in (600, 650, 700): c.line(50, y, 550, y)
        for x, y, text in ((60, 670, 'Role'), (260, 670, 'Responsibility'),
                           (60, 620, 'Clerk'), (260, 620, 'Diagnostic Record')):
            c.drawString(x, y, text)
        c.showPage()
    c.save()
    writer = PdfWriter()
    reader = PdfReader(original)
    for index, rotation in enumerate((0, 0, 90, 180, 270, 0)):
        page = writer.add_page(reader.pages[index])
        if rotation: page.rotate(rotation)
        if index in (3, 5): page.cropbox = RectangleObject([100, 100, 500, 750])
    path = root/'source.pdf'
    with path.open('wb') as stream: writer.write(stream)
    doc = {'sha256': sha256_file(path), 'member': 'own/source.pdf', 'page_count': 6}
    class Browser:
        audit = {'source_archives': {'data': {'sha256': 'own-diagnostic-archive'}}}
        documents = {'own': doc}
        byid = {}
        def checked_page(self, name, page):
            if name != 'own' or isinstance(page, bool) or not isinstance(page, int) or not 1 <= page <= 6:
                raise ValueError('Page outside diagnostic source')
            return doc
        def verified_pdf(self, name):
            if name != 'own': raise ValueError('Unknown document')
            return path
    rows = native_tables(path, doc, 'own', 1, 'own-diagnostic-archive')
    return path, doc, Browser(), rows


class VisualEvidenceTests(unittest.TestCase):
    def test_native_crop_is_exact_source_bound_and_reproduced(self):
        with tempfile.TemporaryDirectory() as root:
            path, doc, browser, rows = fixture(root)
            visual = VisualEvidence(SourceVerifier(browser, rows))
            output = Path(root)/'visual'
            binding = visual.prepare_region(rows[0]['evidence_id'], output)
            self.assertEqual(binding['selection'], 'verified_native_region_crop')
            self.assertEqual(binding['source_bbox_original'], rows[0]['source_bbox'])
            self.assertEqual(binding['source']['doc_sha256'], sha256_file(path))
            self.assertEqual(binding['geometry']['region_pdf_bbox_bottom_left'], [50, 600, 550, 700])
            self.assertEqual(binding['geometry']['source_native_top_left_to_pdf_matrix'], [1, 0, 0, -1, 0, 792])
            with Image.open(output/'page.png') as page, Image.open(output/'crop.png') as cropped:
                self.assertEqual(page.crop(binding['geometry']['crop_page_pixel_bbox_half_open']).tobytes(), cropped.tobytes())
                self.assertLess(cropped.width*cropped.height, page.width*page.height)
            point = [60, 670]
            pixel = transform(binding['geometry']['pdf_to_crop_pixel_matrix'], point)
            restored = transform(binding['geometry']['crop_pixel_to_pdf_matrix'], pixel)
            for a, b in zip(point, restored): self.assertAlmostEqual(a, b)
            self.assertTrue(visual.verify(output)['all_passed'])
            for flag in ('vision_reasoning_performed', 'visual_token_budget_verified', 'answer_generated', 'supported_answer'):
                self.assertFalse(binding[flag])
            with self.assertRaises(ValueError): visual.prepare_region(rows[0]['evidence_id'], output)

    def test_actual_rotate_and_cropbox_maps_land_on_known_colored_pixels(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, browser, _ = fixture(root)
            visual = VisualEvidence(SourceVerifier(browser))
            rotations = (0, 90, 180, 270, 0)
            for page, rotation in zip(range(2, 7), rotations):
                with self.subTest(page=page, rotation=rotation):
                    output = Path(root)/('page-'+str(page))
                    binding = visual.prepare_page('own', page, output, dpi=72)
                    geometry = binding['geometry']
                    self.assertEqual(geometry['page_rotation_clockwise'], rotation)
                    self.assertIsNone(binding['source_bbox_original'])
                    self.assertIsNone(binding['crop_image'])
                    self.assertEqual(binding['fallback_reason'], 'explicit_page_request_region_geometry_missing')
                    with Image.open(output/'page.png') as image:
                        for point, rgb in (([160, 175], (255, 0, 0)), ([330, 330], (0, 255, 0))):
                            pixel = transform(geometry['pdf_to_page_pixel_matrix'], point)
                            self.assertEqual(image.getpixel(tuple(round(v) for v in pixel)), rgb)
                            restored = transform(geometry['page_pixel_to_pdf_matrix'], pixel)
                            for a, b in zip(point, restored): self.assertAlmostEqual(a, b)
                    self.assertTrue(visual.verify(output)['all_passed'])

    def test_cross_page_artifacts_keep_distinct_identity_and_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, browser, _ = fixture(root)
            visual = VisualEvidence(SourceVerifier(browser))
            first = visual.prepare_page('own', 1, Path(root)/'first')
            second = visual.prepare_page('own', 2, Path(root)/'second')
            self.assertEqual(first['source']['doc_sha256'], second['source']['doc_sha256'])
            self.assertNotEqual(first['source']['page_idx'], second['source']['page_idx'])
            self.assertNotEqual(first['binding_id'], second['binding_id'])
            self.assertNotEqual(first['page_image']['sha256'], second['page_image']['sha256'])

    def test_missing_region_and_out_of_bounds_native_record_fail_before_output(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, browser, rows = fixture(root)
            visual = VisualEvidence(SourceVerifier(browser, rows))
            output = Path(root)/'rejected'
            with self.assertRaises(ValueError): visual.prepare_region('missing', output)
            for bbox in (None, [0, 0, 9999, 792]):
                corrupted = copy.deepcopy(rows[0]); corrupted['source_bbox'] = bbox
                with self.assertRaises(ValueError):
                    VisualEvidence(SourceVerifier(browser, [corrupted])).prepare_region(corrupted['evidence_id'], output)
            self.assertFalse(output.exists())

    def test_native_geometry_does_not_claim_rotated_or_cropped_extraction(self):
        with tempfile.TemporaryDirectory() as root:
            path, doc, _, _ = fixture(root)
            for page in (3, 4, 5, 6):
                with self.assertRaises(ValueError): native_tables(path, doc, 'own', page, 'own-diagnostic-archive')

    def test_tampered_png_and_manifest_are_rejected_by_original_rerender(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, browser, rows = fixture(root)
            visual = VisualEvidence(SourceVerifier(browser, rows)); output = Path(root)/'visual'
            visual.prepare_region(rows[0]['evidence_id'], output)
            crop = output/'crop.png'; original = crop.read_bytes(); crop.write_bytes(original+b'tampered')
            self.assertFalse(visual.verify(output)['all_passed']); crop.write_bytes(original)
            manifest = output/'binding.json'; value = json.loads(manifest.read_text(encoding='utf-8'))
            value['source_bbox_original'][0] += 1
            manifest.write_text(json.dumps(value), encoding='utf-8')
            self.assertFalse(visual.verify(output)['all_passed'])

    def test_source_pdf_change_and_page_change_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            path, _, browser, _ = fixture(root)
            visual = VisualEvidence(SourceVerifier(browser)); output = Path(root)/'visual'
            visual.prepare_page('own', 1, output)
            value = json.loads((output/'binding.json').read_text(encoding='utf-8')); value['source']['page'] = 2
            (output/'binding.json').write_text(json.dumps(value), encoding='utf-8')
            self.assertFalse(visual.verify(output)['all_passed'])
            path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaises(ValueError): visual.verify(output)

    def test_bad_page_dpi_bbox_and_fake_verifier_reject(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, browser, _ = fixture(root); visual = VisualEvidence(SourceVerifier(browser))
            for page in (0, 7, True, 1.5):
                with self.assertRaises(ValueError): visual.prepare_page('own', page, Path(root)/'bad')
            for dpi in (True, 35, 145, 100.5):
                with self.assertRaises(ValueError): visual.prepare_page('own', 1, Path(root)/'bad', dpi=dpi)
            for bbox in (None, [0, 0, 0, 1], [2, 2, 1, 1], [-1, 0, 1, 1], [0, 0, float('nan'), 1], [False, 0, 1, 1]):
                with self.assertRaises(ValueError): rectangle(bbox, 612, 792)
            with self.assertRaises(TypeError): VisualEvidence(object())
            self.assertFalse((Path(root)/'bad').exists())

    def test_nonstandard_userunit_refuses_physical_dpi_claim(self):
        with tempfile.TemporaryDirectory() as root:
            path, doc, browser, _ = fixture(root)
            writer = PdfWriter(); writer.append(path)
            writer.pages[0][NameObject('/UserUnit')] = FloatObject(2)
            other = Path(root)/'unit.pdf'
            with other.open('wb') as stream: writer.write(stream)
            path.write_bytes(other.read_bytes()); doc['sha256'] = sha256_file(path)
            with self.assertRaises(ValueError): VisualEvidence(SourceVerifier(browser)).prepare_page('own', 1, Path(root)/'bad')

    def test_raster_budget_rejects_large_real_page_before_render(self):
        with tempfile.TemporaryDirectory() as root:
            path, doc, browser, _ = fixture(root)
            writer = PdfWriter(); writer.append(path)
            writer.pages[0].mediabox = RectangleObject([0, 0, 10000, 10000])
            writer.pages[0].cropbox = RectangleObject([0, 0, 10000, 10000])
            other = Path(root)/'large.pdf'
            with other.open('wb') as stream: writer.write(stream)
            path.write_bytes(other.read_bytes()); doc['sha256'] = sha256_file(path)
            with self.assertRaisesRegex(ValueError, 'render budget'):
                VisualEvidence(SourceVerifier(browser)).prepare_page('own', 1, Path(root)/'bad')


if __name__ == '__main__': unittest.main()
