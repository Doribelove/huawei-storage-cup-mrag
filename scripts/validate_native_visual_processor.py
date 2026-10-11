"""Real pinned CPU image preprocessing; successful validation still exits2 for no generation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
from unittest.mock import patch

from evidence_lab.reader import LocalQwenReader, ReaderUnavailable
from evidence_lab.reader_context import replay_reviewed_visual_context
from evidence_lab.reader_processor import (NativeVisualInputs, PinnedVisualProcessor, VisualPixelPolicy,
    prepare_native_visual_inputs, native_visual_inputs_verified)
from evidence_lab.safe_data import sha256_file


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def negative_live_checks(inputs):
    """Mutate this run's genuine objects only; never alter accepted source files."""
    checks = {}
    clone = NativeVisualInputs(inputs.prepared,inputs.processor,inputs.tensors,inputs.report)
    checks['unregistered_metadata_copy_rejected'] = not native_visual_inputs_verified(clone)
    original = inputs.report['generation_authorized']
    try:
        inputs.report['generation_authorized'] = True
        checks['report_flag_change_rejected'] = not native_visual_inputs_verified(inputs)
    finally: inputs.report['generation_authorized'] = original
    for field in ('input_ids','pixel_values'):
        value = inputs.tensors[field].reshape(-1)
        saved = value[0].clone()
        try:
            value[0] += 1
            checks[field+'_byte_change_rejected'] = not native_visual_inputs_verified(inputs)
        finally: value[0] = saved
    image = inputs.processor._backend.image_processor
    for field in ('min_pixels','size'):
        saved = getattr(image,field)
        try:
            setattr(image,field,saved+1024 if field=='min_pixels' else dict(saved,longest_edge=524289))
            checks[field+'_drift_rejected'] = not native_visual_inputs_verified(inputs)
        finally: setattr(image,field,saved)
    with patch.object(inputs.processor._backend,'apply_chat_template') as fake:
        checks['substituted_native_method_rejected'] = not native_visual_inputs_verified(inputs)
        checks['substituted_method_not_called'] = fake.call_count == 0
    checks['genuine_packet_restored_and_verified'] = native_visual_inputs_verified(inputs)
    if not all(checks.values()): raise ValueError('A genuine live-state rejection regression failed')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processor-dir', type=Path, required=True)
    parser.add_argument('--reviewed-handoff', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    import torch
    from PIL import Image
    from safetensors.torch import save_file
    from transformers import PreTrainedModel
    from transformers.generation.utils import GenerationMixin
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    git = ['git','-c','safe.directory='+str(root.resolve())]
    cases = ('native-commerce-page4','native-crm-page6')
    results = []; started = time.perf_counter()
    with (patch.object(socket.socket,'connect',side_effect=RuntimeError('Offline processor validation')) as network,
          patch.object(socket,'create_connection',side_effect=RuntimeError('Offline processor validation')) as connections,
          patch.object(PreTrainedModel,'from_pretrained',side_effect=RuntimeError('Model loading prohibited')) as models,
          patch.object(GenerationMixin,'generate',side_effect=RuntimeError('Generation prohibited')) as generation):
        try:
            for label,maximum in (('official_bounds_diagnostic',16777216),('bounded_policy',524288)):
                processor = PinnedVisualProcessor(args.processor_dir,policy=VisualPixelPolicy(max_pixels=maximum))
                for case in cases:
                    folder = args.output/(label+'-'+case); folder.mkdir()
                    prepared = replay_reviewed_visual_context(args.reviewed_handoff,case)
                    before = {r['local_image_path']:sha256_file(Path(r['local_image_path']))
                              for r in prepared.citation_catalog.values()}
                    inputs = prepare_native_visual_inputs(prepared,processor)
                    report = inputs.report
                    # Serialize tensor evidence safely. These are image inputs, never weights.
                    tensor_file = folder/'input-tensors.safetensors'
                    save_file(inputs.tensors,str(tensor_file))
                    write(folder/'expanded-input-ids.json',inputs.tensors['input_ids'][0].tolist())
                    live = native_visual_inputs_verified(inputs)
                    single = []; negatives = {}
                    if label == 'bounded_policy':
                        offset = 0
                        for index,record in enumerate(prepared.citation_catalog.values()):
                            with Image.open(record['local_image_path']) as image:
                                image.load()
                                actual = processor._backend.image_processor(images=image,return_tensors='pt')
                            grid = inputs.tensors['image_grid_thw'][index:index+1]
                            rows = int(grid.prod().item())
                            equal = torch.equal(actual['image_grid_thw'],grid) and torch.equal(
                                actual['pixel_values'],inputs.tensors['pixel_values'][offset:offset+rows])
                            offset += rows
                            single.append({'image_id':record['image_id'],'single_image_grid':actual['image_grid_thw'].tolist(),
                                'single_vs_batch_grid_and_pixels_exact':equal,'native_image_processor_run':True})
                        if not all(item['single_vs_batch_grid_and_pixels_exact'] for item in single):
                            raise ValueError('Native single/batch image order or pixel values differ')
                        if not live: raise ValueError('Bounded native packet is not genuinely verified')
                        negatives = negative_live_checks(inputs)
                    reader = LocalQwenReader.__new__(LocalQwenReader)
                    try:
                        reader.read_visual(inputs)
                    except ReaderUnavailable as exc:
                        gate = {'blocked':True,'reason':str(exc)}
                    else: raise ValueError('Model generation must stay blocked')
                    item = {'case':case,'policy':label,'native_inputs_verified':live,
                        'native_validation':report,'single_image_checks':single,'negative_live_state_checks':negatives,
                        'model_generation_gate':gate,
                        'tensor_file':str(tensor_file),'tensor_file_sha256':sha256_file(tensor_file),
                        'source_images_unchanged':before == {r['local_image_path']:sha256_file(Path(r['local_image_path']))
                            for r in prepared.citation_catalog.values()},
                        'original_primary_rank':prepared.report['primary_retrieval_ranks'][0],
                        'original_roles':[r['role'] for r in prepared.citation_catalog.values()],
                        'source_replay':prepared.report['source_preparation_replay']}
                    write(folder/'native-processor-case.json',item); results.append(item)
            bounded = [item for item in results if item['policy']=='bounded_policy']
            passed = len(bounded)==2 and all(item['native_inputs_verified'] and item['source_images_unchanged'] for item in bounded)
            status = 'native_processor_and_visual_budgets_passed_generation_blocked' if passed else 'native_processor_validation_failed'
            code = 2 if passed else 1
        except Exception as exc:
            status = 'native_processor_validation_error'; code = 3
            write(args.output/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        counts = {'network_attempts':network.call_count+connections.call_count,'model_load_attempts':models.call_count,
                  'model_generate_attempts':generation.call_count}
    report = {'status':status,'completed_utc':datetime.now(timezone.utc).isoformat(),
        'source_commit':subprocess.check_output([*git,'rev-parse','HEAD'],cwd=root,text=True).strip(),
        'working_tree_clean':not subprocess.check_output([*git,'status','--porcelain'],cwd=root,text=True).strip(),
        'reviewed_handoff_sha256':sha256_file(args.reviewed_handoff),'cases':results,'prohibited_calls':counts,
        'native_multimodal_preprocessing_cases_completed':len(results),
        'native_single_image_preprocessing_checks':sum(len(item['single_image_checks']) for item in results),
        'source_images_referenced_per_bounded_run':6,'new_images_materialized':0,
        'actual_reader_model_run':False,'answer_accuracy_measured':False,'supported_answer':False,
        'actual_generation_readiness_exit_code':code,'processor_validation_passed':code==2,
        'elapsed_seconds':time.perf_counter()-started,
        'exit_code_contract':'2=native preprocessing validation passed but model generation prohibited;1=validation failure;3=error'}
    if any(counts.values()): raise RuntimeError('A prohibited action was attempted')
    write(args.output/'native-visual-processor-report.json',report)
    print(json.dumps({k:report[k] for k in ('status','source_commit','processor_validation_passed','prohibited_calls',
        'actual_generation_readiness_exit_code')},ensure_ascii=False))
    return code


if __name__ == '__main__': raise SystemExit(main())
