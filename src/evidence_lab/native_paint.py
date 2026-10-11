"""Exclude PDF objects certainly outside Form BBoxes or nonpainting text.

This is a local parser adapter, not a renderer or semantic proof. Explicit path
clipping, alpha, optional content and subsequent occlusion remain unverified.
No installed library is modified; unfiltered source records remain in the audit.
"""
from collections import Counter

from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.utils import apply_matrix_rect, mult_matrix
from pdfplumber.page import Page, PDFPageAggregatorWithMarkedContent


def disjoint(a,b):
    # Lines/points have zero extent on an axis but can still lie inside a Form.
    # Boundary-touching objects are retained; this filter proves exclusion only.
    return min(a[2],b[2])<max(a[0],b[0]) or min(a[3],b[3])<max(a[1],b[1])


class FormBoundDevice(PDFPageAggregatorWithMarkedContent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.form_constraints=[]
        self.text_render_mode=0

    def begin_figure(self,name,bbox,matrix):
        if len(self.form_constraints)>=8:raise ValueError('Native Form nesting budget exceeded')
        # PDF Form BBox is [left,bottom,right,top]. LTFigure currently treats
        # its last two coordinates as widths; keep the actual PDF box here.
        actual=apply_matrix_rect(mult_matrix(matrix,self.ctm),bbox)
        self.form_constraints.append({'name':name,'bbox_pdf_layout':list(actual),
                                      'matrix':list(mult_matrix(matrix,self.ctm))})
        super().begin_figure(name,bbox,matrix)

    def end_figure(self,name):
        super().end_figure(name)
        self.form_constraints.pop()

    def render_string(self,textstate,seq,ncs,graphicstate):
        old=self.text_render_mode
        self.text_render_mode=int(textstate.render)
        try:return super().render_string(textstate,seq,ncs,graphicstate)
        finally:self.text_render_mode=old

    def tag_new(self,start,text=False):
        for obj in self.cur_item._objs[start:]:
            constraints=[dict(x) for x in self.form_constraints]
            obj.source_form_constraints=constraints
            obj.source_outside_form=any(disjoint(obj.bbox,c['bbox_pdf_layout']) for c in constraints)
            obj.source_partial_form=any(not (c['bbox_pdf_layout'][0]<=obj.x0 and obj.x1<=c['bbox_pdf_layout'][2]
                and c['bbox_pdf_layout'][1]<=obj.y0 and obj.y1<=c['bbox_pdf_layout'][3]) for c in constraints)
            if text:obj.source_text_render_mode=self.text_render_mode

    def render_char(self,*args,**kwargs):
        start=len(self.cur_item._objs)
        result=super().render_char(*args,**kwargs)
        self.tag_new(start,text=True)
        return result

    def paint_path(self,*args,**kwargs):
        start=len(self.cur_item._objs)
        super().paint_path(*args,**kwargs)
        self.tag_new(start)

    def render_image(self,*args,**kwargs):
        start=len(self.cur_item._objs)
        super().render_image(*args,**kwargs)
        self.tag_new(start)


class FormBoundPage(Page):
    @property
    def layout(self):
        if not hasattr(self,'_layout'):
            device=FormBoundDevice(self.pdf.rsrcmgr,pageno=self.page_number,laparams=self.pdf.laparams)
            PDFPageInterpreter(self.pdf.rsrcmgr,device).process_page(self.page_obj)
            self._layout=device.get_result()
        return self._layout

    def process_object(self,obj):
        value=super().process_object(obj)
        for key in ('source_form_constraints','source_outside_form','source_partial_form','source_text_render_mode'):
            if hasattr(obj,key):value[key]=getattr(obj,key)
        return value


def exclusion_reason(obj):
    if obj.get('source_outside_form'):return 'outside_actual_transformed_form_bbox'
    if obj.get('object_type')=='char' and obj.get('source_text_render_mode') in (3,7):
        return 'nonpainting_text_render_mode'
    return None


def form_bound_page(original):
    native=FormBoundPage(original.pdf,original.page_obj,original.page_number,original.initial_doctop)
    if len(native.chars)>200000:raise ValueError('Native glyph budget exceeded')
    excluded=[obj for group in native.objects.values() for obj in group if exclusion_reason(obj)]
    removed_chars=[x for x in excluded if x['object_type']=='char']
    from pdfplumber.utils import extract_words
    audit={'schema':'native-form-bbox-paint-filter-v1','input_chars':len(native.chars),
           'excluded_chars':len(removed_chars),'excluded_objects':len(excluded),
           'excluded_reason_counts':dict(Counter(exclusion_reason(x) for x in excluded)),
           'excluded_word_records':extract_words(removed_chars,extra_attrs=['fontname','size']) if removed_chars else [],
           'partial_form_objects_retained':sum(bool(x.get('source_partial_form')) and not exclusion_reason(x)
               for group in native.objects.values() for x in group),
           'unfiltered_records_preserved':True,'semantic_association_verified':False,
           'limitation':'Form AABB disjointness and text modes only; path clipping, alpha, optional content, rotated Form interiors and occlusion unverified'}
    return native.filter(lambda obj:exclusion_reason(obj) is None),native,audit
