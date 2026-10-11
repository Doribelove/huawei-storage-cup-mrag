"""Source-only support transcription for explicitly requested parameter rows.

The original image remains complete. Full native records remain in the source
catalog; this projection is a transcription, never semantic support.
"""
from .native_fields import word_rows
from .parameter_fields import requested_parameter_candidate


def parameter_source_projection(question, validated, cited_catalog):
    literal = validated.get('source_literal_candidate')
    if literal is None:
        return None
    checked = requested_parameter_candidate(question, cited_catalog)
    if checked is None or any(literal.get(key) != checked[key] for key in
            ('schema', 'answer', 'evidence_ids', 'source_field_plan')):
        raise ValueError('Literal support projection requires unchanged native fields')
    projected = {}
    for group in checked['source_field_plan']['groups']:
        source = cited_catalog[group['image_id']]
        rows = group['rows']
        title_top = min(row['title_top'] for row in rows)
        header_bottom = max(w['bottom'] for row in rows for w in row['column_header_words'])
        words = source['readable_viewport']['native_word_records']
        heading = [' '.join(w['text'] for w in line) for y, line in word_rows(words)
                   if title_top - 1 <= y <= header_bottom]
        if group['title'] not in heading:
            raise ValueError('Actual table heading absent from support transcription')
        projected[group['image_id']] = {
            'native_table_heading_and_preface_lines': heading,
            'native_parameter_rows': [{key: row[key] for key in ('row_label', 'columns', 'values')}
                                      for row in rows],
            'native_parameter_projection': {
                'schema': 'requested-native-parameter-support-v1',
                'literal_word_geometry_rechecked': True,
                'complete_source_image_retained': True,
                'full_native_records_retained_in_source_catalog': True,
                'semantic_support_established': False}}
    return projected
