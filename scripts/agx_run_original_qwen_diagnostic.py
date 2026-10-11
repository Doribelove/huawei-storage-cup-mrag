"""Gated, offline original Qwen3-VL-8B execution on four own diagnostic cases.

This program does not retrieve a private corpus, attest semantic evidence support,
enable the product Reader, or measure independent competition answer accuracy.
Run only under the separately bounded task-local original-model supervisor.
"""
import atexit
import hashlib
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = (Path.home() / 'huawei-mrag-runtime-20261007').resolve()
REPORT = ROOT / 'state/original-Qwen-diagnostic.json'


def hashes(path, git_blob=False):
    sha = hashlib.sha256()
    blob = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode()) if git_blob else None
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            sha.update(block)
            if blob is not None:
                blob.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': sha.hexdigest(), 'git_blob_sha1': blob.hexdigest() if blob else None}


def save(report):
    temporary = REPORT.with_suffix('.json.new')
    with temporary.open('w', encoding='utf-8') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n'); output.flush(); os.fsync(output.fileno())
    os.replace(temporary, REPORT)


def own_image(fixture, directory):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (640, 480), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=32)
    if fixture == 'bar_chart':
        draw.text((40, 25), 'Observed counts', font=font, fill='black')
        draw.line((80, 100, 80, 380, 570, 380), fill='black', width=4)
        draw.rectangle((170, 230, 300, 378), fill='#2458a6')
        draw.rectangle((380, 110, 510, 378), fill='#269766')
        for x, y, value in [(205, 188, '5'), (415, 68, '9')]:
            draw.text((x, y), value, font=font, fill='black')
        draw.text((205, 390), 'A', font=font, fill='black')
        draw.text((415, 390), 'B', font=font, fill='black')
        draw.text((85, 430), 'Unit: count', font=font, fill='black')
    elif fixture == 'capacitor_formula':
        for y, value in [(45, 'Capacitor energy'), (150, 'E = (1/2) * C * V^2'),
                         (235, 'C = 2 uF'), (310, 'V = 10 V'), (395, '1 uF = 10^-6 F')]:
            draw.text((40, y), value, font=font, fill='black')
    else:
        raise ValueError('unknown own image fixture')
    path = directory / (fixture + '.png')
    assert not path.exists()
    image.save(path)
    return image, {'source_png_sha256': hashes(path)['sha256'],
                   'RGB_pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                   'original_image_size': list(image.size), 'page': 1,
                   'coordinate_space': 'own original image pixels', 'PDF_source': False,
                   'private_material_uploaded': False}


def main():
    assert platform.machine() == 'aarch64' and sys.version_info[:3] == (3, 12, 14)
    assert Path(sys.prefix).resolve() == (ROOT / 'env').resolve()
    sys.path.insert(0, str(ROOT / 'scripts'))
    from agx_original_qwen_contract import (FIXED_REPOSITORY, FIXED_REVISION, require_real_qualification,
        verify_identity, verify_weight_index, input_messages, validate_candidate, compare_own_reference)
    from agx_download_budget import require_approval
    proof_path = ROOT / 'state/gpu-reader-qualification.json'
    proof = json.loads(proof_path.read_text())
    require_real_qualification(proof)
    plan_path = ROOT / 'scripts/agx-gpu-build-execution.json'
    plan = json.loads(plan_path.read_text())
    require_approval(plan, 'model', proof)
    protocol_path = ROOT / 'scripts/agx-original-qwen-diagnostic.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['model_repository'] == FIXED_REPOSITORY and protocol['model_revision'] == FIXED_REVISION
    assert protocol['attention'] == 'eager' and protocol['dtype'] == 'float16' and protocol['batch_size'] == 1
    assert protocol['maximum_prompt_tokens'] == 4096 and protocol['maximum_new_tokens'] == 192
    lock = ROOT / 'state/original-Qwen-diagnostic.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor); atexit.register(lock.unlink)
    assert not REPORT.exists(), 'preserve previous actual model failures; audit a rerun separately'
    report = {'schema': 'actual-original-Qwen-own-diagnostic-v1', 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': hashes(protocol_path)['sha256'],
        'fixed_resource_plan_sha256': hashes(plan_path)['sha256'], 'qualification_sha256': hashes(proof_path)['sha256'],
        'model_repository': FIXED_REPOSITORY, 'model_revision': FIXED_REVISION,
        'original_model_weights_loaded': False, 'original_tensor_loading_started': False,
        'original_model_CPU_load_completed': False, 'main_product_Reader_generation_enabled': False,
        'independent_accuracy_benchmark': False, 'answer_accuracy_measured': False,
        'semantic_evidence_support': 'unverified', 'supported_answer': False,
        'self_created_evidence_only': True, 'network_connection_attempts': 0,
        'paid_API_calls': 0, 'electricity_or_hardware_cost_measured': False, 'cases': []}

    def deny_network(event, arguments):
        if event == 'socket.connect' and arguments[0].family in (socket.AF_INET, socket.AF_INET6):
            report['network_connection_attempts'] += 1
            raise RuntimeError('original-model diagnostic is offline')

    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    sys.addaudithook(deny_network)
    save(report)
    try:
        resources = [row for row in plan['resources'] if row['id'].startswith('qwen-')]
        assert len(resources) == 12
        identities = {}
        file_stats = {}
        for row in resources:
            path = ROOT / 'downloads' / row['filename']
            assert path.parent == ROOT / 'downloads' and path.is_file() and not path.is_symlink()
            receipt = json.loads((ROOT / 'state' / (row['id'] + '-receipt.json')).read_text())
            before = path.stat()
            observed = hashes(path, bool(row.get('git_blob_sha1')))
            after = path.stat()
            assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
            verify_identity(row, receipt, observed)
            identities[row['filename']] = observed
            file_stats[row['filename']] = (after.st_ino, after.st_size, after.st_mtime_ns)
        verify_weight_index(json.loads((ROOT / 'downloads/model.safetensors.index.json').read_text()), resources)
        report['actual_verified_original_artifacts'] = identities
        save(report)
        model_dir = ROOT / 'models/Qwen3-VL-8B-original-diagnostic'
        model_dir.mkdir(exist_ok=False)
        for row in resources:
            (model_dir / row['filename']).symlink_to(Path('../../downloads') / row['filename'])
        fixtures = ROOT / 'diagnostics/original-Qwen-own-fixtures'
        fixtures.mkdir(parents=True, exist_ok=False)
        import torch
        import torchvision
        import transformers
        from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
        assert torch.__version__.split('+')[0] == '2.3.1' and torchvision.__version__.split('+')[0] == '0.18.1'
        assert transformers.__version__ == '4.57.6' and torch.version.cuda == '11.4' and torch.cuda.is_available()
        assert all(proof['runtime'][name] == value for name, value in
                   [('Python', platform.python_version()), ('torch', torch.__version__),
                    ('torchvision', torchvision.__version__), ('Transformers', transformers.__version__), ('CUDA', torch.version.cuda)])
        torch.set_num_threads(2); torch.manual_seed(protocol['seed'])
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False,
            use_fast=True, min_pixels=protocol['minimum_image_pixels'], max_pixels=protocol['maximum_image_pixels'])
        image_processor = processor.image_processor
        assert image_processor.size == {'shortest_edge': 65536, 'longest_edge': 524288}
        assert image_processor.min_pixels == 65536 and image_processor.max_pixels == 524288
        config = Qwen3VLConfig.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        for part in [config, config.text_config, config.vision_config]:
            part._attn_implementation = 'eager'
        for name, expected in file_stats.items():
            value = (ROOT / 'downloads' / name).stat()
            assert (value.st_ino, value.st_size, value.st_mtime_ns) == expected
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        report['original_tensor_loading_started'] = True
        save(report)
        model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_dir), config=config,
            torch_dtype=torch.float16, attn_implementation='eager', use_safetensors=True,
            local_files_only=True, trust_remote_code=False).eval()
        report['original_model_CPU_load_completed'] = True
        save(report)
        model = model.to('cuda')
        torch.cuda.synchronize()
        assert all(parameter.device.type == 'cuda' and parameter.dtype == torch.float16 for parameter in model.parameters())
        assert model.config.text_config._attn_implementation == model.config.vision_config._attn_implementation == 'eager'
        report.update(original_model_weights_loaded=True, model_class=type(model).__name__,
            native_processor_class=type(processor).__name__, native_image_processor_class=type(image_processor).__name__,
            actual_parameters=sum(parameter.numel() for parameter in model.parameters()),
            model_loading_seconds=time.perf_counter() - started, dtype='float16', attention='eager',
            torch_compile_used=False, model_CPU_offload_used=False, model_quantization_used=False,
            runtime=proof['runtime'], random_seed=protocol['seed'])
        save(report)
        for case in protocol['cases']:
            before = time.perf_counter()
            evidence = case['evidence']
            image, source = own_image(evidence['fixture'], fixtures) if evidence['kind'] == 'image' else (None,
                {'source_text_sha256': hashlib.sha256(evidence['text'].encode()).hexdigest(), 'page': evidence['page'], 'PDF_source': False})
            messages = input_messages(case, image)
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image] if image is not None else None,
                               padding=True, return_tensors='pt')
            tokens = int(inputs['input_ids'].shape[1])
            assert tokens <= protocol['maximum_prompt_tokens']
            expected_visual = 0
            grid = []
            if image is not None:
                grid = inputs['image_grid_thw'].tolist()
                expected_visual = int((inputs['image_grid_thw'].prod(dim=-1) // image_processor.merge_size ** 2).sum())
                image_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
                assert int((inputs['input_ids'] == image_id).sum()) == expected_visual
            preprocessing_seconds = time.perf_counter() - before
            cuda_inputs = {name: tensor.to(device='cuda', dtype=torch.float16 if tensor.is_floating_point() else tensor.dtype)
                           for name, tensor in inputs.items()}
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            before = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**cuda_inputs, max_new_tokens=protocol['maximum_new_tokens'],
                    do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - before
            generated_ids = generated[0, tokens:].detach().cpu()
            assert 0 < len(generated_ids) <= protocol['maximum_new_tokens']
            raw = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            candidate = validate_candidate(raw, evidence, image.size if image else None)
            row = {'case_id': case['id'], 'source_id': evidence['source_id'], 'source': source,
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'reference_labels_in_prompt': False,
                'complete_prompt_tokens': tokens, 'expanded_visual_tokens': expected_visual, 'image_grid_thw': grid,
                'new_tokens': len(generated_ids), 'preprocessing_seconds': preprocessing_seconds,
                'generation_seconds': seconds, 'CUDA_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'CUDA_peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'raw_generated_candidate': raw,
                'output_contract': candidate, 'own_reference_diagnostic': compare_own_reference(candidate, case['reference']),
                'retrieval_executed': False, 'main_product_answer_promoted': False}
            report['cases'].append(row); save(report)
            print(json.dumps({'case_id': case['id'], 'generation_seconds': seconds, 'new_tokens': len(generated_ids),
                              'structural_validity': candidate['structural_validity'], 'semantic_support': 'unverified'}), flush=True)
            del inputs, cuda_inputs, generated
            torch.cuda.empty_cache()
        assert report['network_connection_attempts'] == 0
        report.update(status='completed_runtime_diagnostic', completed_utc=datetime.now(timezone.utc).isoformat())
        save(report)
    except Exception as error:
        report.update(status='stopped', error_type=type(error).__name__,
                      reason=str(error).replace(str(Path.home()), '~'), completed_utc=datetime.now(timezone.utc).isoformat())
        save(report)
        print(json.dumps({'status': 'stopped', 'error_type': type(error).__name__, 'reason': report['reason']}), flush=True)
        return 3
    print(json.dumps({'status': report['status'], 'own_cases_executed': len(report['cases']),
                      'independent_accuracy_benchmark': False, 'main_product_generation_enabled': False}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
