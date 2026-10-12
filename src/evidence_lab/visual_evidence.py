"""Verified local PDF pages/crops; no inferred geometry, OCR or model calls."""
from __future__ import annotations

import hashlib
import importlib.metadata
from io import BytesIO
import json
import math
from pathlib import Path
import threading

from PIL import Image
from pypdf import PdfReader
import pypdfium2 as pdfium

from .native_pdf import KIND, SourceVerifier
from .safe_data import evidence_output, sha256_file

_PDFIUM_LOCK = threading.RLock()
MAX_PIXELS = 6_000_000


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def transform(matrix, point):
    a, b, c, d, e, f = matrix
    x, y = point
    return [a*x+c*y+e, b*x+d*y+f]


def inverse(matrix):
    a, b, c, d, e, f = matrix
    determinant = a*d-b*c
    if not math.isfinite(determinant) or abs(determinant) < 1e-12:
        raise ValueError('Singular page coordinate transform')
    return [d/determinant, -b/determinant, -c/determinant, a/determinant,
            (c*f-d*e)/determinant, (b*e-a*f)/determinant]


def rectangle(value, width, height):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)
            or not (0 <= value[0] < value[2] <= width and 0 <= value[1] < value[3] <= height)):
        raise ValueError('Region rectangle is missing, empty, inverted or outside the original page')
    return list(value)


def image_record(image: Image.Image, name: str):
    stream = BytesIO(); image.save(stream, format='PNG', optimize=False)
    data = stream.getvalue()
    return {'file': name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
            'mode': image.mode, 'size_pixels': list(image.size),
            'rgb_pixels_sha256': hashlib.sha256(image.tobytes()).hexdigest()}, data


class VisualEvidence:
    """Reuse authoritative SourceVerifier IDs and its verified local PDF resolver."""
    def __init__(self, verifier: SourceVerifier):
        if type(verifier) is not SourceVerifier:
            raise TypeError('Visual source needs the existing SourceVerifier')
        self.verifier = verifier
        self.browser = verifier.browser
        self.render_calls = 0

    def _source(self, evidence_id, document_name, page):
        row = None
        proof = None
        if evidence_id is not None:
            row = self.verifier.extra.get(evidence_id, self.browser.byid.get(evidence_id))
            if row is None:
                raise ValueError('Unknown region evidence ID; use an explicit verified page request')
            proof = self.verifier.verify(evidence_id)
            if not proof['all_passed']:
                raise ValueError('Region source verification failed; no crop or page substituted')
            row = evidence_output(row)
            document_name, page = row['document_name'], row['page']
        doc = self.browser.checked_page(document_name, page)
        path = self.browser.verified_pdf(document_name)
        if path.stat().st_size > 32*1024**2 or sha256_file(path) != doc['sha256']:
            raise ValueError('Original PDF bytes changed or exceeded existing read budget')
        if row is not None and (row['doc_sha256'] != doc['sha256'] or row['page_idx'] != page-1):
            raise ValueError('Region PDF/page binding mismatch')
        return path, doc, document_name, page, row, proof

    def _build(self, *, evidence_id=None, document_name=None, page=None, dpi=100, readable_question=None,
            full_page_recovery=False,native_unit_prose_recovery=False,referenced_figure_context=False,native_paragraph_window=False,diagram_structure_view=False):
        if isinstance(dpi, bool) or not isinstance(dpi, int) or not 36 <= dpi <= 144:
            raise ValueError('Visual render DPI must be 36..144')
        path, doc, name, number, row, proof = self._source(evidence_id, document_name, page)
        reader = PdfReader(path)
        if len(reader.pages) != doc['page_count']:
            raise ValueError('Original PDF page count differs from verified document')
        source_page = reader.pages[number-1]
        media = [float(x) for x in source_page.mediabox]
        cropbox = [float(x) for x in source_page.cropbox]
        rotation = int(source_page.get('/Rotate', 0)) % 360
        unit = float(source_page.get('/UserUnit', 1))
        if unit != 1 or rotation not in (0, 90, 180, 270):
            raise ValueError('Unsupported PDF user-unit/rotation; no invented DPI transform')
        readable = None
        if type(full_page_recovery) is not bool or (full_page_recovery and readable_question is None):
            raise ValueError('Recovery needs a boolean flag and the original question')
        if type(native_unit_prose_recovery) is not bool or (native_unit_prose_recovery and not full_page_recovery):
            raise ValueError('Native unit prose is restricted to full original-page recovery')
        if (type(native_paragraph_window) is not bool or (native_paragraph_window and
                (readable_question is None or full_page_recovery or referenced_figure_context))):
            raise ValueError('Native paragraph needs a separate original-source question view')
        if (type(referenced_figure_context) is not bool
                or (referenced_figure_context and (readable_question is None or full_page_recovery))):
            raise ValueError('Linked figure requires a separate original-source question view')
        if (type(diagram_structure_view) is not bool or (diagram_structure_view and
                (readable_question is None or full_page_recovery or referenced_figure_context or native_paragraph_window))):
            raise ValueError('Diagram view requires a separate original figure request')
        if readable_question is not None:
            if row is None: raise ValueError('Readable window must retain an actual retrieved source')
            if rotation == 0 and cropbox == media and media[:2] == [0.0,0.0]:
                from .readable_region import plan_pdf_window
                readable=plan_pdf_window(path,readable_question,row,media[2],media[3],
                    full_page_recovery=full_page_recovery,native_unit_prose_recovery=native_unit_prose_recovery,
                    referenced_figure_context=referenced_figure_context,native_paragraph_window=native_paragraph_window,diagram_structure_view=diagram_structure_view)
            else:
                if referenced_figure_context or native_paragraph_window or diagram_structure_view:
                    raise ValueError('Linked-figure view has unsupported original page transform')
                readable={'status':'full_page_fallback','question':readable_question,
                    'reason':'unsupported_native_window_page_transform','association_verified':False}
            dpi=readable.get('dpi',100)
        with _PDFIUM_LOCK, pdfium.PdfDocument(path) as pdf:
            pdfpage = pdf[number-1]
            try:
                if pdfpage.get_rotation() != rotation:
                    raise ValueError('PDF parsers disagree on rotation')
                width, height = pdfpage.get_size()
                if (not all(math.isfinite(v) and 0 < v <= 10000 for v in (width, height))
                        or math.ceil(width*dpi/72)*math.ceil(height*dpi/72) > MAX_PIXELS):
                    raise ValueError('Page raster exceeds bounded render budget')
                bitmap = pdfpage.render(scale=dpi/72, rotation=0, crop=(0, 0, 0, 0),
                    may_draw_forms=False, draw_annots=True)
                self.render_calls += 1
                try:
                    image = bitmap.to_pil().convert('RGB').copy()
                    converter = bitmap.get_posconv(pdfpage)
                    w, h = image.size
                    p0 = converter.to_page(0, 0); px = converter.to_page(w, 0); py = converter.to_page(0, h)
                    pixel_to_pdf = [(px[0]-p0[0])/w, (px[1]-p0[1])/w,
                                    (py[0]-p0[0])/h, (py[1]-p0[1])/h, p0[0], p0[1]]
                    pdf_to_pixel = inverse(pixel_to_pdf)
                    visible = list(pdfpage.get_bbox())
                    corners = [(visible[0], visible[1]), (visible[2], visible[1]),
                               (visible[2], visible[3]), (visible[0], visible[3])]
                    checkpoints = []
                    for point in corners:
                        continuous = transform(pdf_to_pixel, point)
                        native = list(converter.to_bitmap(*point))
                        if max(abs(a-b) for a,b in zip(continuous, native)) > 1.01:
                            raise ValueError('Continuous transform disagrees with native PDFium mapping')
                        checkpoints.append({'pdf_point': list(point), 'pixel_continuous': continuous,
                                            'native_pixel_rounded': native})
                finally:
                    bitmap.close()
            finally:
                pdfpage.close()
        page_asset, page_bytes = image_record(image, 'page.png')
        reason = 'explicit_page_request_region_geometry_missing'
        pdf_region = None
        pixel_box = None
        original_box = row.get('source_bbox') if row else None
        native_geometry = row is not None and row.get('extraction', {}).get('kind') == KIND
        if native_geometry:
            # native_tables itself rejects rotated/cropped pages. Also require an
            # origin-zero media box before using its top-left PDF point convention.
            if rotation != 0 or cropbox != media or media[:2] != [0.0, 0.0]:
                reason = 'native_geometry_page_transform_not_supported_use_full_visible_page'
            elif row.get('bbox_coordinate_system') != 'PDF native points; top-left; pdfplumber':
                reason = 'native_coordinate_convention_missing_use_full_visible_page'
            else:
                bbox = rectangle(original_box, media[2], media[3])
                if row.get('page_size_points') != [media[2], media[3]]:
                    raise ValueError('Native region page-size association mismatch')
                pdf_region = [bbox[0], media[3]-bbox[3], bbox[2], media[3]-bbox[1]]
                projected = [transform(pdf_to_pixel, p) for p in ((pdf_region[0],pdf_region[1]),
                    (pdf_region[2],pdf_region[1]),(pdf_region[2],pdf_region[3]),(pdf_region[0],pdf_region[3]))]
                # One pixel of outward raster sampling covers native integer rounding;
                # source geometry itself is never clamped, padded or rewritten.
                pixel_box = [max(0, math.floor(min(p[0] for p in projected))-1),
                             max(0, math.floor(min(p[1] for p in projected))-1),
                             min(image.width, math.ceil(max(p[0] for p in projected))+1),
                             min(image.height, math.ceil(max(p[1] for p in projected))+1)]
                rectangle(pixel_box, image.width, image.height)
                reason = None
        elif row is not None:
            reason = 'quantized_or_unverified_region_mapping_use_full_visible_page'
        if readable and readable.get('full_page_recovery'):
            # A native region has already populated pixel_box above. Recovery
            # must clear that old crop before binding the full original page.
            pdf_region=None
            pixel_box=None
            reason='bounded_recovery_adds_original_page_outside_first_crop'
        elif readable and readable['status']=='selected_native_pdf_window':
            bbox=rectangle(readable['viewport_bbox_top_left_points'],media[2],media[3])
            pdf_region=[bbox[0],media[3]-bbox[3],bbox[2],media[3]-bbox[1]]
            projected=[transform(pdf_to_pixel,p) for p in ((pdf_region[0],pdf_region[1]),
                (pdf_region[2],pdf_region[1]),(pdf_region[2],pdf_region[3]),(pdf_region[0],pdf_region[3]))]
            pixel_box=[max(0,math.floor(min(p[0] for p in projected))-1),
                max(0,math.floor(min(p[1] for p in projected))-1),
                min(image.width,math.ceil(max(p[0] for p in projected))+1),
                min(image.height,math.ceil(max(p[1] for p in projected))+1)]
            rectangle(pixel_box,image.width,image.height)
            reason=None
        crop_asset = None
        pdf_to_crop = None
        native_to_pdf = None
        files = {'page.png': page_bytes}
        if pixel_box is not None:
            cropped = image.crop(tuple(pixel_box))
            crop_asset, crop_bytes = image_record(cropped, 'crop.png')
            files['crop.png'] = crop_bytes
            native_to_pdf = [1, 0, 0, -1, 0, media[3]]
            pdf_to_crop = list(pdf_to_pixel)
            pdf_to_crop[4] -= pixel_box[0]
            pdf_to_crop[5] -= pixel_box[1]
        identity = {'document_name': name, 'doc_sha256': doc['sha256'], 'pdf_member': doc['member'],
                    'page': number, 'page_idx': number-1, 'region_evidence_id': evidence_id}
        binding = {'kind': 'local_verified_pdf_visual_binding', 'source': identity,
            'source_archive_sha256': self.browser.audit['source_archives']['data']['sha256'],
            'region_record_sha256': json_hash(row) if row is not None else None,
            'region_source_verification': proof, 'source_bbox_original': original_box,
            'source_coordinate_system': row.get('bbox_coordinate_system') if row else None,
            'geometry': {'media_box_pdf_bottom_left': media, 'crop_box_pdf_bottom_left': cropbox,
                'visible_box_pdf_bottom_left': visible, 'page_rotation_clockwise': rotation, 'user_unit': unit,
                'pdf_to_page_pixel_matrix': pdf_to_pixel, 'page_pixel_to_pdf_matrix': pixel_to_pdf,
                'source_native_top_left_to_pdf_matrix': native_to_pdf,
                'pdf_to_crop_pixel_matrix': pdf_to_crop,
                'crop_pixel_to_pdf_matrix': inverse(pdf_to_crop) if pdf_to_crop else None,
                'matrix_convention': 'X=a*x+c*y+e; Y=b*x+d*y+f; pixel origin top-left',
                'native_conversion_checkpoints': checkpoints, 'region_pdf_bbox_bottom_left': pdf_region,
                'crop_page_pixel_bbox_half_open': pixel_box, 'outward_raster_margin_pixels': 1 if pixel_box else 0},
            'render': {'scale_pixels_per_pdf_canvas_unit': dpi/72, 'dpi_at_user_unit1': dpi,
                'additional_rotation': 0, 'visible_page_not_hidden_media_content': True,
                'annotations_rendered': True, 'interactive_form_fields_rendered': False,
                'document_has_acroform': '/AcroForm' in reader.trailer['/Root'],
                'pypdfium2_version': importlib.metadata.version('pypdfium2'),
                'pdfium_build': str(pdfium.PDFIUM_INFO),
                'pypdf_version': importlib.metadata.version('pypdf'),
                'pillow_version': importlib.metadata.version('pillow')},
            'page_image': page_asset, 'crop_image': crop_asset,
            'selected_image': crop_asset if crop_asset is not None else page_asset,
            'selection': 'verified_native_region_crop' if crop_asset else 'full_visible_pdf_page',
            'fallback_reason': reason, 'source_geometry_fabricated': False,
            'vision_reasoning_performed': False, 'visual_token_budget_verified': False,
            'answer_generated': False, 'supported_answer': False}
        if readable is not None:
            binding['readable_viewport']=readable
            binding['selection']='question_selected_same_page_pdf_window' if pixel_box else 'full_visible_pdf_page'
        binding['binding_id'] = json_hash(binding)
        if sha256_file(path) != doc['sha256']:
            raise ValueError('Original PDF changed during rendering')
        return binding, files

    def prepare_region(self, evidence_id: str, output: Path, *, dpi=100, readable_question=None,
            full_page_recovery=False,native_unit_prose_recovery=False,referenced_figure_context=False,native_paragraph_window=False,diagram_structure_view=False):
        return self._save(output, evidence_id=evidence_id, dpi=dpi, readable_question=readable_question,
            full_page_recovery=full_page_recovery,native_unit_prose_recovery=native_unit_prose_recovery,
            referenced_figure_context=referenced_figure_context,native_paragraph_window=native_paragraph_window,diagram_structure_view=diagram_structure_view)

    def prepare_page(self, name: str, page: int, output: Path, *, dpi=100):
        return self._save(output, document_name=name, page=page, dpi=dpi)

    def _save(self, output: Path, **request):
        output = Path(output)
        if output.exists():
            raise ValueError('Use a new visual artifact directory; preserve existing images')
        binding, files = self._build(**request)
        output.mkdir(parents=True, exist_ok=False)
        for name, data in files.items():
            (output/name).write_bytes(data)
        (output/'binding.json').write_text(json.dumps(binding, ensure_ascii=False, indent=2,
            allow_nan=False)+'\n', encoding='utf-8')
        return binding

    def verify(self, output: Path):
        output = Path(output)
        stored = json.loads((output/'binding.json').read_text(encoding='utf-8'))
        source = stored['source']
        expected, files = self._build(evidence_id=source['region_evidence_id'],
            document_name=source['document_name'], page=source['page'], dpi=stored['render']['dpi_at_user_unit1'],
            readable_question=stored.get('readable_viewport',{}).get('question'),
            full_page_recovery=stored.get('readable_viewport',{}).get('full_page_recovery',False),
            native_unit_prose_recovery=stored.get('readable_viewport',{}).get('native_unit_prose_recovery',False),
            referenced_figure_context=stored.get('readable_viewport',{}).get('referenced_figure_context',False),
            native_paragraph_window=stored.get('readable_viewport',{}).get('native_paragraph_window',False),
            diagram_structure_view=stored.get('readable_viewport',{}).get('diagram_structure_view',False))
        checks = {'source_geometry_manifest_reproduced': expected == stored,
                  'page_png_bytes_reproduced': (output/'page.png').read_bytes() == files['page.png'],
                  'crop_png_bytes_reproduced': ((output/'crop.png').read_bytes() == files['crop.png'])
                    if 'crop.png' in files else not (output/'crop.png').exists()}
        return {'binding_id': stored['binding_id'], 'checks': checks, 'all_passed': all(checks.values()),
                'meaning': 'Original PDF/source/raster/crop binding only; no visual reasoning or answer support'}
