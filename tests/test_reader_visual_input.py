"""Real self-authored PDF -> BM25 -> images -> Reader dry-run, no native model."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from evidence_lab.lexical import BM25
from evidence_lab.reader import (LocalQwenReader, ReaderUnavailable, VisualOutputInvalid, prepare_reader_visual_inputs,
                                 validate_visual_reader_output)
from evidence_lab.reader_context import (ReaderBudget, VisualReaderContext,
    visual_reader_context_verified, PinnedReaderTokenizer, TokenizerUnavailable)
from evidence_lab.table_context import TableContext
from test_table_context import fixture


def prepare_fixture(root):
    visual, primary, binding = fixture(root, known_neighbors=True)
    hits = BM25(list(visual.verifier.extra.values())).search('Body A', 5)
    eid = binding['source']['region_evidence_id']
    contexts = Path(root)/'table-context'; TableContext(visual).prepare(primary, contexts)
    kwargs = {'visual': visual, 'primary_artifacts': {eid: primary}, 'table_context_artifacts': {eid: contexts}}
    prepared = prepare_reader_visual_inputs('Which source labels are visible?', hits, **kwargs)
    return prepared, hits, eid, kwargs


class ReaderVisualInputTests(unittest.TestCase):
    def test_compact_prompt_preserves_original_catalog_and_verified_binding(self):
        with tempfile.TemporaryDirectory() as root:
            original, hits, eid, kwargs = prepare_fixture(root)
            compact = prepare_reader_visual_inputs('Which source labels are visible?', hits,
                **kwargs, prompt_projection='compact_table_cells_v1')
            self.assertEqual(compact.citation_catalog, original.citation_catalog)
            self.assertTrue(visual_reader_context_verified(compact))
            model_text = '\n'.join(c['text'] for c in compact.messages[1]['content'] if c['type'] == 'text')
            for record in compact.citation_catalog.values():
                self.assertNotIn(record['doc_sha256'], model_text)
                self.assertIn(record['image_id'], model_text)
            self.assertIn('citation_template', model_text)
            self.assertEqual(compact.report['primary_retrieval_ranks'], original.report['primary_retrieval_ranks'])
            raw = json.dumps({'status':'not_enough_evidence','answer':'','citations':[]})
            self.assertFalse(validate_visual_reader_output(raw, compact)['supported_answer'])
            copied = VisualReaderContext(copy.deepcopy(compact.messages), copy.deepcopy(compact.citation_catalog), copy.deepcopy(compact.report))
            self.assertFalse(visual_reader_context_verified(copied))

    def test_actual_bm25_to_typed_local_messages_preserves_rank_and_candidate_roles(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, hits, eid, kwargs = prepare_fixture(root)
            before = copy.deepcopy(hits)
            again = prepare_reader_visual_inputs('Which source labels are visible?', hits, **kwargs)
            self.assertEqual(hits, before)
            self.assertTrue(visual_reader_context_verified(prepared))
            self.assertEqual(prepared.report['primary_retrieval_ranks'], [next(i+1 for i,h in enumerate(hits) if h['evidence_id'] == eid)])
            self.assertFalse(prepared.report['retrieval_order_modified'])
            self.assertEqual(list(prepared.citation_catalog), list(again.citation_catalog))
            images = [c for c in prepared.messages[1]['content'] if c['type'] == 'image']
            self.assertEqual(len(images), 3)
            self.assertTrue(all(set(image) == {'type', 'image'} and Path(image['image']).is_file() for image in images))
            self.assertEqual([r['role'] for r in prepared.citation_catalog.values()], ['primary_retrieved_region', 'candidate_context', 'candidate_context'])
            for record in prepared.citation_catalog.values():
                self.assertFalse(record['table_continuity_verified'])
                self.assertFalse(record['header_assigned_to_primary_table'])
                self.assertTrue(record['pdf_to_image_pixel_matrix'])
            self.assertEqual(prepared.report['preparation']['status'], 'source_verified_messages_prepared')
            self.assertFalse(prepared.report['budget_validation']['token_budget_verified'])
            self.assertIsNone(prepared.report['budget_validation']['expanded_prompt_tokens'])
            self.assertEqual(prepared.report['generation']['status'], 'blocked')

    def test_both_visual_and_text_entrypoints_block_without_processor_or_model_calls(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, _, _, _ = prepare_fixture(root)
            reader = LocalQwenReader.__new__(LocalQwenReader); reader.processor = Mock(); reader.model = Mock()
            with self.assertRaises(ReaderUnavailable): reader.read_visual(prepared)
            with self.assertRaises(ReaderUnavailable): reader.read('Question', prepared)
            reader.processor.apply_chat_template.assert_not_called(); reader.model.generate.assert_not_called()

    def test_copied_or_caller_promoted_ready_packet_cannot_authorize_generation(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, _, _, _ = prepare_fixture(root)
            reader = LocalQwenReader.__new__(LocalQwenReader); reader.processor = Mock(); reader.model = Mock()
            copied = VisualReaderContext(copy.deepcopy(prepared.messages), copy.deepcopy(prepared.citation_catalog), copy.deepcopy(prepared.report))
            self.assertFalse(visual_reader_context_verified(copied))
            with self.assertRaises(ReaderUnavailable): reader.read_visual(copied)
            prepared.report['ready_for_reader'] = True
            prepared.report['token_budget_verified'] = True
            prepared.report['budget_validation']['expanded_prompt_tokens'] = 1
            self.assertFalse(visual_reader_context_verified(prepared))
            with self.assertRaises(ReaderUnavailable): reader.read_visual(prepared)
            reader.processor.apply_chat_template.assert_not_called(); reader.model.generate.assert_not_called()

    def test_original_image_change_invalidates_live_preparation(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, _, _, _ = prepare_fixture(root)
            record = next(iter(prepared.citation_catalog.values())); path = Path(record['local_image_path'])
            path.write_bytes(path.read_bytes()+b'changed')
            self.assertFalse(visual_reader_context_verified(prepared))
            with self.assertRaises(ReaderUnavailable): validate_visual_reader_output(json.dumps({'status':'not_enough_evidence','answer':'','citations':[]}), prepared)

    def test_citation_roles_and_native_ids_are_derived_from_verified_image_catalog(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, _, _, _ = prepare_fixture(root)
            records = list(prepared.citation_catalog.values()); primary, header = records[0], records[-1]
            raw = {'status':'answered','answer':'Synthetic contract observation only', 'citations':[
                {'type':'text_quote','image_id':primary['image_id'],'quote':'Body A'},
                {'type':'text_quote','image_id':header['image_id'],'quote':'Item'}]}
            result = validate_visual_reader_output(json.dumps(raw), prepared)
            bindings = result['diagnostic_candidate_output']['citation_bindings']
            self.assertEqual([c['role'] for c in bindings], ['primary_retrieved_region','candidate_context'])
            self.assertIsNone(bindings[-1]['region_evidence_id'])
            self.assertTrue(bindings[-1]['candidate_native_evidence_id'])
            self.assertEqual(bindings[-1]['page'], 1)
            self.assertFalse(result['supported_answer'])
            self.assertFalse(bindings[-1]['header_assigned_to_primary_table'])
            self.assertEqual(result['answer'], '')
            self.assertEqual(result['status'], 'not_enough_evidence')
            self.assertEqual(result['generation_provenance'], 'not_established_by_output_validation')

    def test_unknown_quote_and_model_supplied_roles_fail_distinct_validation_stages(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, _, _, _ = prepare_fixture(root); records = list(prepared.citation_catalog.values())
            primary = records[0]
            for citation, stage in (
                ({'type':'text_quote','image_id':'unknown','quote':'Body A'}, 'source_binding'),
                ({'type':'text_quote','image_id':primary['image_id'],'quote':'fabricated'}, 'source_binding'),
                ({'type':'text_quote','image_id':primary['image_id'],'quote':'Body A','role':'primary_retrieved_region'}, 'structure')):
                with self.assertRaises(VisualOutputInvalid) as caught:
                    validate_visual_reader_output(json.dumps({'status':'answered','answer':'fixture','citations':[citation]}), prepared)
                self.assertEqual(caught.exception.stage, stage)
            refusal = validate_visual_reader_output(json.dumps({'status':'not_enough_evidence','answer':'','citations':[]}), prepared)
            self.assertFalse(refusal['answer_candidate_present'])

    def test_same_table_and_header_assertions_are_withheld_without_semantic_string_detection(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, _, _, _ = prepare_fixture(root); primary, _, header = prepared.citation_catalog.values()
            for answer in ('This is definitely the same table and Item is its header.',
                           '这些列毫无疑问属于当前表格。', 'An innocent looking but unverified answer.'):
                for citations in ([{'type':'text_quote','image_id':header['image_id'],'quote':'Item'}],
                                  [{'type':'text_quote','image_id':primary['image_id'],'quote':'Body A'},
                                   {'type':'text_quote','image_id':header['image_id'],'quote':'Item'}]):
                    result = validate_visual_reader_output(json.dumps({'status':'answered','answer':answer,'citations':citations}), prepared)
                    self.assertEqual(result['structure_validation']['status'], 'passed')
                    self.assertEqual(result['source_binding_validation']['status'], 'passed')
                    self.assertEqual(result['evidence_support']['status'], 'unverified')
                    self.assertFalse(result['evidence_support']['table_continuity_verified'])
                    self.assertFalse(result['evidence_support']['header_assigned_to_primary_table'])
                    self.assertFalse(result['evidence_support']['semantic_verifier_run'])
                    self.assertEqual((result['status'],result['answer'],result['citations']), ('not_enough_evidence','',[]))
                    self.assertEqual(result['presentation_status'], 'withheld_unverified_candidate')
                    self.assertEqual(result['diagnostic_candidate_output']['model_declared_output']['answer'], answer)
                    self.assertTrue(result['diagnostic_candidate_output']['not_for_answer_presentation'])
                    self.assertFalse(result['supported_answer'])

    def test_pure_visual_observation_binds_real_pixels_pdf_page_and_region_without_quote(self):
        from evidence_lab.visual_evidence import transform
        with tempfile.TemporaryDirectory() as root:
            prepared, _, _, _ = prepare_fixture(root); primary = next(iter(prepared.citation_catalog.values()))
            observation = 'A claimed visual feature, deliberately absent from source text.'
            self.assertNotIn(observation, primary['source_text_for_exact_quote'])
            box = [1, 1, 10, 10]
            citation = {'type':'visual_observation','image_id':primary['image_id'],'observation':observation,
                'region':{'coordinate_system':'selected_image_pixels_top_left','bbox_half_open':box}}
            result = validate_visual_reader_output(json.dumps({'status':'answered','answer':'Unverified visual claim','citations':[citation]}), prepared)
            binding = result['diagnostic_candidate_output']['citation_bindings'][0]
            self.assertNotIn('quote', binding)
            self.assertEqual(binding['image']['sha256'], primary['image']['sha256'])
            self.assertEqual((binding['doc_sha256'],binding['page'],binding['region_evidence_id']),
                             (primary['doc_sha256'],primary['page'],primary['region_evidence_id']))
            polygon = binding['observation_region']['pdf_polygon_bottom_left_canvas_points']
            for actual, expected in zip(polygon, ((1,1),(10,1),(10,10),(1,10))):
                back = transform(binding['pdf_to_image_pixel_matrix'], actual)
                for a,b in zip(back, expected): self.assertAlmostEqual(a,b)
            self.assertFalse(binding['observation_validated'])
            self.assertFalse(binding['observation_region']['fact_localization_verified'])
            self.assertFalse(result['evidence_support']['visual_observations_verified'])
            self.assertEqual(result['answer'], '')

    def test_image_only_original_pdf_and_empty_parsed_text_need_no_fabricated_quote(self):
        import hashlib
        import zipfile
        from reportlab.pdfgen import canvas
        from evidence_lab.demo import EvidenceBrowser
        from evidence_lab.native_pdf import SourceVerifier
        from evidence_lab.safe_data import sha256_file
        from evidence_lab.visual_evidence import VisualEvidence
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); pdf = root/'own-image.pdf'
            c = canvas.Canvas(str(pdf), pagesize=(200,200))
            c.setFillColorRGB(1,0,0); c.circle(100,100,35,fill=1,stroke=0); c.showPage(); c.save()
            source = {'type':'image','page_idx':0,'bbox':[0,0,1000,1000],'image_caption':[]}
            parsed = json.dumps([source]).encode(); member = 'own-content.json'
            archive = root/'own.zip'
            with zipfile.ZipFile(archive,'w') as zipped:
                zipped.write(pdf,'own.pdf'); zipped.writestr(member,parsed)
            row = {'role':'source_pdf_region','evidence_id':hashlib.sha256(parsed).hexdigest(),
                'document_name':'own','doc_sha256':sha256_file(pdf),'page':1,'page_idx':0,'type':'image',
                'source_bbox':source['bbox'],'bbox_normalized':[0,0,1,1],'text':'',
                'source_archive_sha256':sha256_file(archive),
                'aliases':[{'document_name':'own','pdf_member':'own.pdf','content_list_member':member,'record_index':0}]}
            audit=root/'audit'; phase=root/'phase'; audit.mkdir(); phase.mkdir()
            regions=audit/'regions.jsonl'; regions.write_text(json.dumps(row)+'\n',encoding='utf-8')
            metadata={'source_archives':{'data':{'path':str(archive),'sha256':sha256_file(archive)}},
                'evidence_corpus_sha256':sha256_file(regions),'dataset':{'documents':{'own':{
                    'sha256':row['doc_sha256'],'member':'own.pdf','page_count':1,
                    'content_list':{'sha256':hashlib.sha256(parsed).hexdigest()}}}}}
            (audit/'data-audit.json').write_text(json.dumps(metadata),encoding='utf-8')
            (phase/'page-review.json').write_text('[]',encoding='utf-8')
            (phase/'integrity-report.json').write_text('{}',encoding='utf-8')
            browser=EvidenceBrowser(audit,phase); visual=VisualEvidence(SourceVerifier(browser))
            artifact=root/'visual'; visual.prepare_region(row['evidence_id'],artifact)
            prepared=prepare_reader_visual_inputs('Describe the visible shape',[row],visual=visual,
                primary_artifacts={row['evidence_id']:artifact})
            image=next(iter(prepared.citation_catalog.values())); width,height=image['image']['size_pixels']
            self.assertEqual(image['source_text_for_exact_quote'],'')
            citation={'type':'visual_observation','image_id':image['image_id'],'observation':'A red circle is visible.',
                'region':{'coordinate_system':'selected_image_pixels_top_left','bbox_half_open':[0,0,width,height]}}
            result=validate_visual_reader_output(json.dumps({'status':'answered','answer':'A red circle.','citations':[citation]}),prepared)
            self.assertEqual(result['source_binding_validation']['status'],'passed')
            self.assertFalse(result['supported_answer'])
            self.assertEqual(result['answer'],'')
            bad={'type':'text_quote','image_id':image['image_id'],'quote':'A red circle'}
            with self.assertRaises(VisualOutputInvalid) as caught:
                validate_visual_reader_output(json.dumps({'status':'answered','answer':'A red circle.','citations':[bad]}),prepared)
            self.assertEqual(caught.exception.stage,'source_binding')

    def test_visual_region_boundaries_and_source_overrides_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, _, _, _ = prepare_fixture(root); primary = next(iter(prepared.citation_catalog.values()))
            base = {'type':'visual_observation','image_id':primary['image_id'],'observation':'Claim',
                'region':{'coordinate_system':'selected_image_pixels_top_left','bbox_half_open':[0,0,10,10]}}
            width,height = primary['image']['size_pixels']
            for box, stage in (([-1,0,10,10],'source_binding'),([0,0,width+1,height],'source_binding'),
                               ([0,0,0,10],'source_binding'),([0,0,True,10],'structure'),
                               ([0,0,1.5,10],'structure')):
                citation = copy.deepcopy(base); citation['region']['bbox_half_open'] = box
                with self.assertRaises(VisualOutputInvalid) as caught:
                    validate_visual_reader_output(json.dumps({'status':'answered','answer':'Claim','citations':[citation]}), prepared)
                self.assertEqual(caught.exception.stage, stage)
            for key,value in (('page',99),('image_sha256','fake'),('supported_answer',True),('quote','fabricated')):
                citation = {**base,key:value}
                with self.assertRaises(VisualOutputInvalid) as caught:
                    validate_visual_reader_output(json.dumps({'status':'answered','answer':'Claim','citations':[citation]}), prepared)
                self.assertEqual(caught.exception.stage, 'structure')

    def test_candidate_visual_observations_keep_their_own_page_and_do_not_grant_header_support(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, _, _, _ = prepare_fixture(root); candidate = list(prepared.citation_catalog.values())[-1]
            width,height = candidate['image']['size_pixels']
            citation = {'type':'visual_observation','image_id':candidate['image_id'],
                'observation':'This is definitely the header of the next page.',
                'region':{'coordinate_system':'selected_image_pixels_top_left','bbox_half_open':[0,0,width,height]}}
            result = validate_visual_reader_output(json.dumps({'status':'answered','answer':'Same table','citations':[citation]}), prepared)
            binding = result['diagnostic_candidate_output']['citation_bindings'][0]
            self.assertEqual(binding['page'],1)
            self.assertIsNone(binding['region_evidence_id'])
            self.assertEqual(binding['candidate_native_evidence_id'], candidate['candidate_native_evidence_id'])
            self.assertFalse(result['source_binding_validation']['primary_cited'])
            self.assertFalse(result['evidence_support']['supplementary_context_confers_table_or_header_support'])
            self.assertEqual(result['answer'],'')

    def test_refusal_and_caller_supplied_support_flags_use_the_same_presentation_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, _, _, _ = prepare_fixture(root)
            raw = {'status':'not_enough_evidence','answer':'','citations':[]}
            result = validate_visual_reader_output(json.dumps(raw), prepared)
            self.assertEqual(result['presentation_status'],'reader_refused')
            self.assertEqual(result['evidence_support']['status'],'no_answer_candidate')
            for key in ('supported_answer','evidence_support','presentation_status'):
                with self.assertRaises(VisualOutputInvalid): validate_visual_reader_output(json.dumps({**raw,key:True}), prepared)
            with self.assertRaises(VisualOutputInvalid):
                validate_visual_reader_output('{"status":"answered","status":"not_enough_evidence","answer":"","citations":[]}', prepared)

    def test_canonical_retrieval_tampering_and_labels_in_omitted_hits_reject(self):
        with tempfile.TemporaryDirectory() as root:
            _, hits, eid, kwargs = prepare_fixture(root)
            wrong = copy.deepcopy(hits); wrong[0]['text'] += 'fabricated'
            with self.assertRaises(ValueError): prepare_reader_visual_inputs('Question', wrong, **kwargs)
            wrong = copy.deepcopy(hits); next(h for h in wrong if h['evidence_id'] != eid)['extraction']['gt_answer'] = 'label'
            with self.assertRaises(ValueError): prepare_reader_visual_inputs('Question', wrong, **kwargs)

    def test_wrong_primary_artifact_and_promoted_context_reject(self):
        with tempfile.TemporaryDirectory() as root:
            _, hits, eid, kwargs = prepare_fixture(root)
            visual = kwargs['visual']; other = next(h for h in hits if h['evidence_id'] != eid)
            other_dir = Path(root)/'other'; visual.prepare_region(other['evidence_id'], other_dir)
            changed = {**kwargs, 'primary_artifacts': {eid: other_dir}, 'table_context_artifacts': {}}
            with self.assertRaises(ValueError): prepare_reader_visual_inputs('Question', hits, **changed)
            file = kwargs['table_context_artifacts'][eid]/'table-context.json'; value = json.loads(file.read_text(encoding='utf-8'))
            value['candidate_context'][0]['header_assigned_to_primary_table'] = True
            file.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaises(ValueError): prepare_reader_visual_inputs('Question', hits, **kwargs)

    def test_missing_retrieved_primary_character_budget_and_fake_tokenizer_reject(self):
        with tempfile.TemporaryDirectory() as root:
            _, hits, eid, kwargs = prepare_fixture(root)
            with self.assertRaises(ValueError): prepare_reader_visual_inputs('Question', [h for h in hits if h['evidence_id'] != eid], **kwargs)
            with self.assertRaises(ValueError): prepare_reader_visual_inputs('Question', hits, budget=ReaderBudget(max_characters=20), **kwargs)
            fake = PinnedReaderTokenizer.__new__(PinnedReaderTokenizer); fake._tokenizer = Mock()
            with self.assertRaises(TokenizerUnavailable): prepare_reader_visual_inputs('Question', hits, tokenizer=fake, **kwargs)
            fake._tokenizer.apply_chat_template.assert_not_called()

    def test_multiple_primary_contexts_fail_aggregate_image_budget_without_truncation(self):
        with tempfile.TemporaryDirectory() as root:
            _, hits, eid, kwargs = prepare_fixture(root)
            visual = kwargs['visual']; second = next(h for h in hits if h['page'] == 2)
            folder = Path(root)/'second-primary'; visual.prepare_region(second['evidence_id'], folder)
            context = Path(root)/'second-context'; TableContext(visual).prepare(folder, context)
            changed = {**kwargs,'primary_artifacts':{eid:kwargs['primary_artifacts'][eid],second['evidence_id']:folder},
                'table_context_artifacts':{eid:kwargs['table_context_artifacts'][eid],second['evidence_id']:context}}
            with self.assertRaisesRegex(ValueError, 'aggregate budget'):
                prepare_reader_visual_inputs('Question', hits, **changed)


if __name__ == '__main__': unittest.main()
