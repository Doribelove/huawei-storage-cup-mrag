"""Question-independent image-region search features from original PDF words.

This is acquisition metadata. A nearby caption or a word rectangle does not
establish a depicted object's identity, a semantic cell, or answer support.
Canonical source records are never rewritten with this derived search text.
"""
import hashlib
import json
import re

from .readable_region import native_word_lines, valid_box


def _word_box(word):
    return [word['x0'],word['top'],word['x1'],word['bottom']]


def _inside(word,box):
    return (box[0]<=word['x0'] and word['x1']<=box[2]
            and box[1]<=word['top'] and word['bottom']<=box[3])


def _bounded_groups(words,limit):
    selected=[];used=0
    for groups in native_word_lines(words):
        line=' | '.join(groups)
        if used+len(line)+1>limit:
            break
        selected.append(line);used+=len(line)+1
    return '\n'.join(selected)


def image_region_features(source,page_words,*,width,height,page_record_sha256):
    """Keep word rectangles and spatial caption candidates, without semantics."""
    if source.get('type')!='image':
        raise ValueError('Image region required')
    if len(page_words)>15000:
        raise ValueError('Original page word budget exceeded')
    box=source.get('bbox_points_estimate')
    if not valid_box(box,width,height):
        raise ValueError('Original image region estimate is outside the page')
    words=[]
    for word in page_words:
        if not isinstance(word.get('text'),str) or not valid_box(_word_box(word),width,height):
            raise ValueError('Native word record is outside the actual page')
        words.append({k:word[k] for k in ('text','x0','x1','top','bottom')})
    inside=[word for word in words if _inside(word,box)]
    # Keep surrounding native lines as search cues. Never assert that an
    # adjacent caption describes a particular raster object or flow node.
    surrounding=[word for word in words if not _inside(word,box)
        and box[0]<=word['x0'] and word['x1']<=box[2]
        and ((box[1]-64<=word['top'] and word['bottom']<=box[1])
             or (box[3]<=word['top'] and word['bottom']<=box[3]+64))]
    captions=[]
    for groups in native_word_lines(surrounding):
        line=' | '.join(groups)
        if re.match(r'^\s*(?:fig(?:ure)?\.?|table|chart)\s*\d',line,re.I):
            captions.append(line)
    literal_inside=_bounded_groups(inside,4000)
    context=_bounded_groups(surrounding,1600)
    original=source.get('text','')
    feature_text='\n'.join(value for value in (original,literal_inside,context) if value)
    identity={key:source[key] for key in ('evidence_id','document_name','doc_sha256','page')}
    return {
        'schema':'native-image-region-search-features-v1',**identity,
        'original_source_record_sha256':hashlib.sha256(json.dumps(source,ensure_ascii=False,
            sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'page_record_sha256':page_record_sha256,
        'source_bbox_coordinate_system':source['bbox_coordinate_system'],
        'source_bbox_points_estimate':list(box),
        'native_words_in_region':inside,
        'nearby_native_words_for_search_only':surrounding,
        'nearby_caption_candidates_for_search_only':captions,
        'native_region_word_count':len(inside),
        'original_parsed_image_text':original,
        'search_feature_text':feature_text,
        'search_text_has_native_region_words':bool(inside),
        'no_native_region_text_visual_acquisition_needed':not bool(inside),
        'canonical_source_text_replaced':False,
        'source_quote_or_answer_fact_established':False,
        'caption_is_answer_sentence':False,
        'semantic_cell_or_graph_node_association_verified':False,
        'question_or_reference_labels_used':False,
        'limits':['Quantized original parsed region box; actual native word coordinates retained.',
            'Nearby captions/context are retrieval cues only; not depicted object or node assertions.',
            'Native Form/nonpainting filter is partial visibility analysis; source pixels remain necessary.',
            'Search text is bounded at whole spatial lines; complete word records remain in the page artifact.'],
    }
