"""The compact citation contract binds identity, never answer semantics."""
import copy
import json
import tempfile
import unittest

from evidence_lab.reader import (bind_compact_source_citations, prepare_reader_visual_inputs,
    validate_visual_reader_output, VisualOutputInvalid, ReaderUnavailable)
from test_reader_visual_input import prepare_fixture


class CompactCitationTests(unittest.TestCase):
    def candidate(self, image_id, **extra):
        return {'status':'answered', 'answer':'A deliberately unverified claim',
            'citations':[{'image_id':image_id, 'observation':'A claimed visible fact', **extra}]}

    def test_projection_preserves_claim_and_full_source_binding_but_never_releases(self):
        with tempfile.TemporaryDirectory() as root:
            original, hits, _, kwargs = prepare_fixture(root)
            prepared = prepare_reader_visual_inputs('Which source labels are visible?', hits,
                prompt_projection='compact_table_cells_and_citations_v2', **kwargs)
            self.assertEqual(prepared.citation_catalog, original.citation_catalog)
            source = next(iter(prepared.citation_catalog.values()))
            raw = self.candidate(source['image_id'])
            result = bind_compact_source_citations(json.dumps(raw), prepared)
            self.assertEqual(result['diagnostic_candidate_output']['model_declared_output'], raw)
            effective = result['diagnostic_candidate_output']['effective_candidate_output']
            self.assertEqual(effective['answer'], raw['answer'])
            self.assertEqual(effective['citations'][0]['region']['bbox_half_open'],
                [0,0,*source['image']['size_pixels']])
            binding = result['diagnostic_candidate_output']['citation_bindings'][0]
            self.assertEqual(binding['image']['sha256'], source['image']['sha256'])
            self.assertEqual(binding['doc_sha256'], source['doc_sha256'])
            self.assertFalse(binding['observation_region']['fact_localization_verified'])
            self.assertFalse(binding['observation_region']['model_declared_geometry'])
            self.assertEqual(binding['observation_region']['selection_provenance'],
                'full_already_selected_image_by_application_contract')
            self.assertTrue(result['answer_candidate_present'])
            self.assertEqual(result['answer'], '')
            self.assertFalse(result['supported_answer'])
            self.assertFalse(result['evidence_support']['semantic_verifier_run'])
            self.assertFalse(result['compact_citation_projection']['answer_changed'])
            from evidence_lab.support import finalize_answer
            forged = finalize_answer(result, {'accepted':True,'semantic_verifier_run':True})
            self.assertEqual(forged['answer'], '')
            self.assertFalse(forged['supported_answer'])
            with self.assertRaises(VisualOutputInvalid):
                validate_visual_reader_output(json.dumps(raw), prepared)

    def test_unknown_duplicate_extra_fields_and_malformed_outputs_reject(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, _, _, _ = prepare_fixture(root)
            image_id = next(iter(prepared.citation_catalog))
            duplicate = self.candidate(image_id)
            duplicate['citations'].append(copy.deepcopy(duplicate['citations'][0]))
            cases = [self.candidate('unknown'), duplicate,
                self.candidate(image_id, supported_answer=True),
                self.candidate(image_id, role='primary_retrieved_region'),
                self.candidate(image_id, type='text_quote'),
                self.candidate(image_id, type='visual_observation'),
                self.candidate(image_id, region={}), self.candidate(image_id, observation='')]
            for value in cases:
                with self.subTest(value=value), self.assertRaises(VisualOutputInvalid):
                    bind_compact_source_citations(json.dumps(value), prepared)
            for raw in ('{"status":"answered","answer":',
                '{"status":"answered","status":"not_enough_evidence","answer":"","citations":[]}'):
                with self.assertRaises(VisualOutputInvalid):
                    bind_compact_source_citations(raw, prepared)

    def test_typed_illegal_geometry_stale_packet_and_refusal_payload_still_reject(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, _, _, _ = prepare_fixture(root)
            image_id = next(iter(prepared.citation_catalog))
            raw = self.candidate(image_id, type='visual_observation', region={
                'coordinate_system':'selected_image_pixels_top_left', 'bbox_half_open':[-1,0,10,10]})
            with self.assertRaises(VisualOutputInvalid):
                bind_compact_source_citations(json.dumps(raw), prepared)
            refusal = {'status':'not_enough_evidence','answer':'','citations':[]}
            self.assertEqual(bind_compact_source_citations(json.dumps(refusal), prepared)['presentation_status'], 'reader_refused')
            refusal['citations'] = self.candidate(image_id)['citations']
            with self.assertRaises(VisualOutputInvalid):
                bind_compact_source_citations(json.dumps(refusal), prepared)
            prepared.citation_catalog[image_id]['image']['size_pixels'][0] += 1
            with self.assertRaises(ReaderUnavailable):
                bind_compact_source_citations(json.dumps(self.candidate(image_id)), prepared)


    def test_model_declared_legal_bbox_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, _, _, _ = prepare_fixture(root)
            image_id = next(iter(prepared.citation_catalog))
            box = [1,1,10,10]
            raw = self.candidate(image_id, type='visual_observation', region={
                'coordinate_system':'selected_image_pixels_top_left', 'bbox_half_open':box})
            result = bind_compact_source_citations(json.dumps(raw), prepared)
            effective = result['diagnostic_candidate_output']['effective_candidate_output']
            self.assertEqual(effective, raw)
            self.assertEqual(result['diagnostic_candidate_output']['citation_bindings'][0]
                ['observation_region']['bbox_half_open'], box)
            self.assertEqual(result['compact_citation_projection']['derived_regions'], [])
            self.assertFalse(result['supported_answer'])


if __name__ == '__main__': unittest.main()
