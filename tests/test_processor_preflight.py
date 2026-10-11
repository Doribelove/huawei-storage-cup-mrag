"""Automation/configuration contracts only; fixtures prove no native processor parity."""
import copy
from types import SimpleNamespace
import unittest

from scripts.validate_reader_processor import processor_exit_code, summarize_image_component


class ProcessorPreflightContractTests(unittest.TestCase):
    def complete_text_metadata(self):
        comparison = {'status': 'compared', 'token_ids_equal': True, 'rendered_sha256_equal': True,
                      'within_prompt_budget': True, 'within_prompt_plus_generation_budget': True}
        return {'status': 'text_processor_tokens_verified', 'actual_text_processor_token_equality': True,
                'investigation_complete': True, 'construction': {'status': 'constructed', 'class': 'Qwen3VLProcessor'},
                'calls': {'native_processor_constructed': 1},
                'cases': [{'processor_text_comparison': copy.deepcopy(comparison)} for _ in range(2)]}

    def test_completed_investigation_with_unavailable_processor_is_nonzero(self):
        report = {'status': 'processor_unavailable', 'investigation_complete': True,
                  'actual_text_processor_token_equality': None, 'calls': {'native_processor_constructed': 0}}
        self.assertEqual(processor_exit_code(report), 2)

    def test_success_requires_complete_strict_two_case_text_proof(self):
        good = self.complete_text_metadata()
        self.assertEqual(processor_exit_code(good), 0)
        variants = []
        for field, value in [('actual_text_processor_token_equality', 'true'), ('investigation_complete', False)]:
            bad = copy.deepcopy(good); bad[field] = value; variants.append(bad)
        bad = copy.deepcopy(good); bad['calls']['native_processor_constructed'] = 0; variants.append(bad)
        bad = copy.deepcopy(good); bad['construction']['class'] = 'SyntheticProcessor'; variants.append(bad)
        bad = copy.deepcopy(good); bad['cases'] = []; variants.append(bad)
        bad = copy.deepcopy(good); bad['cases'][1]['processor_text_comparison']['token_ids_equal'] = False; variants.append(bad)
        bad = copy.deepcopy(good); bad['cases'][1]['processor_text_comparison']['within_prompt_plus_generation_budget'] = False; variants.append(bad)
        for bad in variants:
            with self.subTest(report=bad): self.assertEqual(processor_exit_code(bad), 1)

    def test_unknown_mismatched_or_missing_status_cannot_pass(self):
        for status in (None, 'text_processor_tokens_mismatch', 'investigation_complete'):
            self.assertEqual(processor_exit_code({'status': status}), 1)

    def image_metadata(self, size, minimum, maximum):
        return SimpleNamespace(size=size, min_pixels=minimum, max_pixels=maximum,
                               patch_size=16, temporal_patch_size=2, merge_size=2)

    def test_loader_pixel_attributes_do_not_hide_effective_size_drift(self):
        expected = {'size': {'shortest_edge': 65536, 'longest_edge': 16777216},
                    'patch_size': 16, 'temporal_patch_size': 2, 'merge_size': 2}
        component = self.image_metadata({'shortest_edge': 3136, 'longest_edge': 1003520}, 65536, 16777216)
        report = summarize_image_component(component, expected)
        self.assertFalse(report['size_matches_pinned_config'])
        self.assertFalse(report['pixel_attributes_agree_with_effective_size'])
        self.assertFalse(report['pinned_pixel_and_patch_settings_match'])
        self.assertFalse(report['visual_token_budget_verified'])
        component.min_pixels = 3136; component.max_pixels = 1003520
        report = summarize_image_component(component, expected)
        self.assertTrue(report['pixel_attributes_agree_with_effective_size'])
        self.assertFalse(report['pinned_pixel_and_patch_settings_match'])

    def test_synchronized_metadata_still_does_not_prove_visual_tokens(self):
        expected = {'size': {'shortest_edge': 65536, 'longest_edge': 524288},
                    'patch_size': 16, 'temporal_patch_size': 2, 'merge_size': 2}
        for minimum, maximum in ((65536, 524288), (None, None)):
            component = self.image_metadata(copy.deepcopy(expected['size']), minimum, maximum)
            report = summarize_image_component(component, expected)
            self.assertTrue(report['pinned_pixel_and_patch_settings_match'])
            self.assertFalse(report['image_processing_performed'])
            self.assertFalse(report['visual_token_budget_verified'])


if __name__ == '__main__': unittest.main()
