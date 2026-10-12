import copy
import json
import tempfile
import unittest

from evidence_lab.reader import prepare_reader_visual_inputs
from evidence_lab.reader_context import visual_reader_context_verified
from test_reader_visual_input import prepare_fixture


class QuestionPlacementTests(unittest.TestCase):
    def test_original_sources_images_and_question_survive_registered_reordering(self):
        with tempfile.TemporaryDirectory() as root:
            _, hits, _, kwargs = prepare_fixture(root)
            question = 'Which source labels are visible?'
            before = prepare_reader_visual_inputs(question, hits, **kwargs,
                prompt_projection='answer_source_ids_v3')
            after = prepare_reader_visual_inputs(question, hits, **kwargs,
                prompt_projection='answer_source_ids_v3', question_after_images=True)
            self.assertEqual(before.messages[0], after.messages[0])
            self.assertEqual(before.citation_catalog, after.citation_catalog)
            old = copy.deepcopy(before.messages[1]['content'])
            payload = json.loads(old[0]['text'])
            self.assertEqual(payload.pop('question'), question)
            self.assertEqual(json.loads(after.messages[1]['content'][0]['text']), payload)
            self.assertEqual(after.messages[1]['content'][1:-1], old[1:])
            self.assertEqual(after.messages[1]['content'][-1],
                             {'type': 'text', 'text': 'Question: ' + question})
            self.assertTrue(visual_reader_context_verified(after))
            # The new layout is bound at preparation; a later caller cannot
            # move/change question text without invalidating the live context.
            after.messages[1]['content'][-1]['text'] += ' changed'
            self.assertFalse(visual_reader_context_verified(after))
            self.assertTrue(visual_reader_context_verified(before))

    def test_invalid_layout_requests_do_not_create_live_contexts(self):
        with tempfile.TemporaryDirectory() as root:
            _, hits, _, kwargs = prepare_fixture(root)
            for value in ('true', 1, None):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    prepare_reader_visual_inputs('Describe the source', hits, **kwargs,
                        prompt_projection='answer_source_ids_v3', question_after_images=value)
            with self.assertRaises(ValueError):
                prepare_reader_visual_inputs('Describe the source', hits, **kwargs,
                    question_after_images=True)
