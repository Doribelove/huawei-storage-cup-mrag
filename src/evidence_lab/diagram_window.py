"""Whole original figure and adjacent native lines, without semantic labels."""
import math
from .readable_region import valid_box, native_word_lines
from .same_page_figure import native_lines

DISPLAY_QUESTION='Display the entire original diagram with its original adjacent caption.'

def diagram_window(question, source, words, width, height):
    if question != DISPLAY_QUESTION:
        raise ValueError('Diagram display request must be question independent')
    box=source.get('bbox_points_estimate')
    if (source.get('type')!='image' or not valid_box(box,width,height)
            or source.get('bbox_coordinate_system')!='MinerU integer 0..1000, top-left; PDF points estimate is quantized'):
        raise ValueError('Diagram display requires canonical original figure geometry')
    viewport=[max(0,box[0]-18),max(0,box[1]-18),min(width,box[2]+18),min(height,box[3]+18)]
    bounded=[];excluded=[]
    for word in words:
        extent=[word[k] for k in ('x0','top','x1','bottom')]
        if any(type(n) not in (int,float) or not math.isfinite(n) for n in extent):
            raise ValueError('Invalid original word geometry')
        (bounded if valid_box(extent,width,height) else excluded).append(word)
    context=[]
    for line in native_lines(bounded):
        e=line['bbox_top_left_points']
        if not (e[2]>=box[0]-18 and e[0]<=box[2]+18
                and (box[1]-48<=e[3]<=box[1] or box[3]<=e[1]<=box[3]+48)):
            continue
        context.append({**line,'is_figure_labels_or_answer_sentence':False})
        viewport=[max(0,min(viewport[0],e[0]-6)),max(0,min(viewport[1],e[1]-6)),
            min(width,max(viewport[2],e[2]+6)),min(height,max(viewport[3],e[3]+6))]
    visible=[w for w in bounded if viewport[0]<=w['x0'] and w['x1']<=viewport[2]
             and viewport[1]<=w['top'] and w['bottom']<=viewport[3]]
    return {'status':'selected_native_pdf_window','policy':'original_complete_diagram_window_v1',
        'question':question,'kind':'original_quantized_figure_region','diagram_structure_view':True,
        'native_object_bbox_top_left_points':list(box),'viewport_bbox_top_left_points':viewport,
        'margin_pdf_points':18,'original_context_lines':context,
        'native_word_records':[{k:w[k] for k in ('text','x0','top','x1','bottom')} for w in visible],
        'native_visible_text':' '.join(w['text'] for w in visible),
        'native_context_text':' '.join(l['text'] for l in context),
        'native_spatial_word_lines':native_word_lines(visible),'spatial_word_groups_are_semantic_cells':False,
        'native_page_word_geometry_audit':{'excluded_nonvisible_word_records':excluded,
            'excluded_nonvisible_word_count':len(excluded),'word_coordinates_altered':False},
        'dpi':144 if math.ceil(width*2)*math.ceil(height*2)<=6_000_000 else 100,
        'source_bbox_precision':'original quantized figure extent, not verified nodes or label rectangles',
        'entire_original_figure_extent_retained':True,'association_verified':False,
        'fact_localization_verified':False,'table_continuity_verified':False,
        'source_caption_is_answer_sentence':False,'reference_pages_used':False}
