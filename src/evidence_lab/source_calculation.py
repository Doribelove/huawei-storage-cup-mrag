"""Source-cell arithmetic candidates, kept separate from original model output.

Only a unique, labelled native count matrix and explicitly requested classes
are eligible. This constructs a candidate; unchanged source support still
decides presentation. There are no benchmark identifiers or answer constants.
"""
import copy
import hashlib
import json
import re

from .confusion_fields import confusion_rate_plan
from .reader_context import visual_reader_context_verified


def requested_matrix_candidate(question, catalog):
    if re.search(r'\bF[- ]?1\b|\bprecision\b|\brecall\b', question, re.I):
        return None
    plan = confusion_rate_plan(question, catalog)
    if plan is None:
        return None
    labels = plan['row_labels']
    # Bind the question's named categories, not a dataset modality label.
    if not all(re.search(r'(?<![\w-])'+re.escape(label)+r'(?![\w-])', question,
                         0 if len(label)==1 else re.I) for label in labels):
        return None
    reference = next(group for group in (plan['row_conditioned_rates'], plan['column_conditioned_rates'])
                     if group[0]['conditioning_axis']=='reference')
    prediction = next(group for group in (plan['row_conditioned_rates'], plan['column_conditioned_rates'])
                      if group[0]['conditioning_axis']=='prediction')
    parts = []
    for rate in reference:
        cells = [cell for cell in plan['source_observed_off_diagonal_cells']
                 if cell['reference_class']==rate['class_label']]
        destinations = ' and '.join('predicted '+c['predicted_class']+' in '+c['literal_matrix_value']+' cases'
                                    for c in cells)
        parts.append('Reference '+rate['class_label']+' was '+destinations+
                     ' (reference-conditioned error '+rate['fraction']+' = '+rate['percent_rounded_2dp']+'%)')
    parts.append('Conditioning on predicted category instead: '+', '.join(
        rate['class_label']+' '+rate['fraction']+' = '+rate['percent_rounded_2dp']+'%'
        for rate in prediction))
    answer = '; '.join(parts)+'. Rates are calculated from off-diagonal counts divided by the labelled axis totals.'
    if len(answer)>2400:
        return None
    return {'schema':'source-calculated-matrix-candidate-v1', 'answer':answer,
            'evidence_ids':[plan['image_id']], 'source_field_plan':plan,
            'question_classes_explicitly_matched':list(labels),
            'origin':'deterministic source-cell arithmetic; not Qwen Reader text',
            'semantic_support_established':False}


def compose_source_calculated_candidate(question, generated, prepared):
    if not visual_reader_context_verified(prepared):
        raise ValueError('Source calculation requires a live unchanged source context')
    candidate = requested_matrix_candidate(question, prepared.citation_catalog)
    if candidate is None:
        return generated
    return _compose_source_candidate(generated, prepared, candidate)


def compose_source_parameter_candidate(question, generated, prepared):
    if not visual_reader_context_verified(prepared):
        raise ValueError('Source parameter extraction requires a live unchanged source context')
    from .parameter_fields import requested_parameter_candidate
    candidate = requested_parameter_candidate(question, prepared.citation_catalog)
    if candidate is None:
        return generated
    return _compose_source_candidate(generated, prepared, candidate, field='source_literal_candidate',
        projection_origin='source-native literal row candidate; original Reader bytes separately preserved')


def compose_source_period_candidate(question, generated, prepared):
    if not visual_reader_context_verified(prepared):
        raise ValueError('Period comparison requires a live unchanged source context')
    from .period_comparison import requested_period_candidate
    candidate = requested_period_candidate(question, prepared.citation_catalog)
    if candidate is None:
        return generated
    return _compose_source_candidate(generated, prepared, candidate, field='source_period_candidate',
        projection_origin='source-native period comparison; original Reader bytes separately preserved')


def _compose_source_candidate(generated, prepared, candidate, *, field='source_calculated_candidate',
                              projection_origin='source-calculated candidate; original Reader bytes separately preserved',
                              native_quote_question=None):
    from .reader import bind_answer_source_ids
    if native_quote_question is None:
        bound = bind_answer_source_ids(json.dumps(
            {'answer':candidate['answer'], 'evidence_ids':candidate['evidence_ids']},
            ensure_ascii=False), prepared)
    else:
        from .native_paragraph_quote import FIELD,bind_native_paragraph_quote
        if field!=FIELD:raise ValueError('Native quote binding requires its explicit application contract')
        bound=bind_native_paragraph_quote(native_quote_question,candidate,prepared)
    composed = copy.deepcopy(generated)
    composed.update(bound)
    for raw_field in ('raw_output_for_validation', 'generated_token_ids', 'input_token_ids',
                  'generation_provenance', 'actual_reader_run', 'reader_identity',
                  'input_tokens', 'output_tokens', 'wall_ms', 'hit_token_cap'):
        if raw_field in generated:
            composed[raw_field] = copy.deepcopy(generated[raw_field])
    original_diagnostic = generated.get('diagnostic_candidate_output') or {}
    composed['diagnostic_candidate_output']['model_declared_output'] = copy.deepcopy(
        original_diagnostic.get('model_declared_output', {}))
    composed['diagnostic_candidate_output']['original_reader_binding'] = copy.deepcopy(original_diagnostic)
    composed['diagnostic_candidate_output']['effective_candidate_origin'] = candidate['origin']
    composed[field] = candidate
    composed[field]['original_reader_output_sha256'] = hashlib.sha256(
        generated.get('raw_output_for_validation', '').encode()).hexdigest()
    composed['answer_source_id_projection'].update(
        application_candidate_payload_sha256=composed['answer_source_id_projection']['raw_output_sha256'],
        raw_output_sha256=composed[field]['original_reader_output_sha256'],
        answer_changed=(candidate['answer'] != original_diagnostic.get('model_declared_output',{}).get('answer','')),
        raw_model_output_preserved=True,
        semantic_support_established_by_projection=False,
        projection_origin=projection_origin)
    return composed
