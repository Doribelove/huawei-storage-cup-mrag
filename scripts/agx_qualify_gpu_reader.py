"""Actual private-runtime CUDA, random tiny Qwen and native processor gates, no weights."""
import hashlib
import atexit
import importlib.metadata
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.home() / 'huawei-mrag-runtime-20261007'
REPORT = ROOT / 'state/gpu-reader-qualification.json'
PLAN = json.loads((ROOT / 'scripts/agx-gpu-build-execution.json').read_text())
report = {'schema': 'actual-agx-no-pretrained-weight-qualification-v1',
          'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'steps': [],
          'CUDA_operators_passed': False, 'tiny_Qwen_text_passed': False,
          'tiny_Qwen_multimodal_passed': False, 'native_processor_and_runtime_requalified': False,
          'pretrained_weights_loaded': False, 'answer_accuracy_measured': False,
          'self_created_images_only': True, 'attention': 'eager', 'torch_compile_used': False}
report['network_connection_attempts'] = 0


def deny_network_during_no_weight_gates(event, arguments):
    if event == 'socket.connect' and arguments[0].family in (socket.AF_INET, socket.AF_INET6):
        report['network_connection_attempts'] += 1
        raise RuntimeError('No-weight qualification prohibits network connections')
    if event == 'open' and isinstance(arguments[0], (str, bytes)):
        name = os.fsdecode(arguments[0])
        if name.endswith('.safetensors'):
            raise RuntimeError('No-weight qualification prohibits opening pretrained tensor shards')


def save():
    temporary = REPORT.with_suffix('.json.new')
    with temporary.open('w', encoding='utf-8') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n'); output.flush(); os.fsync(output.fileno())
    os.replace(temporary, REPORT)


def file_sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): value.update(block)
    return value.hexdigest()


def step(name, function):
    before = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    row = {'name': name, 'status': 'passed', 'elapsed_seconds': time.perf_counter() - before, **(result or {})}
    report['steps'].append(row)
    save()
    print(json.dumps(row), flush=True)


def cuda_operators():
    details = []
    for dtype in [torch.float16, torch.bfloat16]:
        left = torch.randn(64, 64, device='cuda', dtype=dtype)
        product = left @ left.T
        normalized = torch.nn.functional.layer_norm(product, (64,))
        probability = torch.softmax(normalized, dim=-1)
        conv = torch.nn.Conv3d(3, 8, kernel_size=(2, 4, 4), stride=(2, 4, 4)).cuda().to(dtype)
        inputs = torch.randn(1, 3, 2, 16, 16, device='cuda', dtype=dtype)
        pixels = conv(inputs)
        assert all(torch.isfinite(value).all().item() for value in [product, normalized, probability, pixels])
        reference = left.float().cpu() @ left.float().cpu().T
        assert torch.allclose(product.float().cpu(), reference, rtol=0.03, atol=0.35)
        norm_reference = torch.nn.functional.layer_norm(product.float().cpu(), (64,))
        assert torch.allclose(normalized.float().cpu(), norm_reference, rtol=0.03, atol=0.03)
        assert torch.allclose(probability.float().cpu(), torch.softmax(normalized.float().cpu(), dim=-1), rtol=0.03, atol=0.01)
        conv_reference = torch.nn.functional.conv3d(inputs.float().cpu(), conv.weight.float().cpu(), conv.bias.float().cpu(), stride=(2, 4, 4))
        assert torch.allclose(pixels.float().cpu(), conv_reference, rtol=0.04, atol=0.03)
        details.append({'dtype': str(dtype), 'matmul_layernorm_softmax_conv3d_finite': True,
                        'CPU_float32_numerical_references_passed': True,
                        'matmul_max_absolute_difference': float((product.float().cpu()-reference).abs().max().item()),
                        'conv3d_max_absolute_difference': float((pixels.float().cpu()-conv_reference).abs().max().item())})
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=torch.float32, device='cuda')
    scores = torch.tensor([0.9, 0.8, 0.7], device='cuda')
    assert torchvision.ops.nms(boxes, scores, 0.5).tolist() == [0, 2]
    resized = torchvision.transforms.functional.resize(torch.rand(3, 64, 96, device='cuda'), [32, 48], antialias=True)
    assert resized.shape == (3, 32, 48) and torch.isfinite(resized).all().item()
    report['CUDA_operators_passed'] = True
    return {'operators': details, 'torchvision_CUDA_nms_and_resize': True}


def native_processor():
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, AutoTokenizer
    directory = ROOT / 'models/qwen3-vl-8b-instruct-0c351dd-processor-only'
    directory.mkdir(exist_ok=True)
    identities = {}
    for resource in PLAN['resources']:
        if resource['kind'] != 'Qwen_JSON': continue
        source = ROOT / 'downloads' / resource['filename']
        data = source.read_bytes()
        assert len(data) == resource['bytes'] and hashlib.sha256(data).hexdigest() == resource['sha256']
        destination = directory / source.name
        if destination.exists(): assert destination.read_bytes() == data
        else: destination.write_bytes(data)
        identities[source.name] = {'bytes': len(data), 'sha256': resource['sha256']}
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False,
        use_fast=True, min_pixels=65536, max_pixels=524288,
        size={'shortest_edge': 65536, 'longest_edge': 524288})
    image_processor = processor.image_processor
    assert type(processor).__module__.startswith('transformers.models.qwen3_vl.')
    assert type(image_processor).__module__.startswith('transformers.models.qwen3_vl.')
    assert image_processor.min_pixels == 65536 and image_processor.max_pixels == 524288
    assert image_processor.size['shortest_edge'] == 65536 and image_processor.size['longest_edge'] == 524288
    text_counts = []
    for text in ['请根据证据回答：这个图中的数值是多少？', 'Use only the cited page and refuse unsupported conclusions.']:
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        expected = tokenizer(prompt, return_tensors='pt')['input_ids']
        actual = processor(text=prompt, return_tensors='pt')['input_ids']
        assert torch.equal(actual, expected)
        text_counts.append(int(actual.numel()))
    images = []
    for index, size in enumerate([(128, 128), (256, 192), (512, 384), (384, 512), (960, 640), (640, 960)]):
        image = Image.new('RGB', size, (235-index*9, 225, 210+index*5))
        drawing = ImageDraw.Draw(image)
        drawing.rectangle((10, 10, size[0]-10, size[1]-10), outline=(10, 30, 70), width=3)
        drawing.text((20, 20), 'SELF-CREATED '+str(index), fill=(0, 0, 0))
        images.append(image)
    messages = [{'role': 'user', 'content': [*({'type': 'image'} for _ in images),
                 {'type': 'text', 'text': 'Describe these six self-created diagnostic images.'}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=prompt, images=images, return_tensors='pt')
    grid = batch['image_grid_thw']
    assert tuple(grid.shape) == (6, 3)
    offset = 0; single_checks = []
    for index, image in enumerate(images):
        single = image_processor(images=image, return_tensors='pt')
        rows = int(grid[index].prod().item())
        assert torch.equal(single['image_grid_thw'], grid[index:index+1])
        difference = float((single['pixel_values'] - batch['pixel_values'][offset:offset+rows]).abs().max().item())
        assert torch.allclose(single['pixel_values'], batch['pixel_values'][offset:offset+rows], atol=1e-6, rtol=1e-6)
        offset += rows
        single_checks.append({'image': index, 'original_size': list(image.size), 'grid': grid[index].tolist(),
                              'single_batch_max_absolute_difference': difference})
    assert offset == batch['pixel_values'].shape[0]
    expected_tokens = int((grid.prod(dim=-1) // image_processor.merge_size ** 2).sum().item())
    image_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    assert int((batch['input_ids'] == image_id).sum().item()) == expected_tokens
    assert batch['input_ids'].numel() <= 8192
    report['native_processor_and_runtime_requalified'] = True
    return {'native_processor_class': type(processor).__name__, 'native_image_processor_class': type(image_processor).__name__,
            'two_complete_text_token_parity_counts': text_counts, 'six_single_batch_image_checks': single_checks,
            'expanded_visual_tokens': expected_tokens, 'complete_prompt_tokens': int(batch['input_ids'].numel()),
            'fixed_public_JSON_identities': identities, 'private_dataset_images_uploaded': False}


def tiny_qwen():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    config = Qwen3VLConfig(text_config={'vocab_size': 256, 'hidden_size': 128, 'intermediate_size': 256,
        'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2, 'head_dim': 32,
        'max_position_embeddings': 256, 'rope_scaling': {'rope_type': 'default', 'mrope_section': [6, 5, 5]},
        'eos_token_id': 2, 'pad_token_id': 0},
        vision_config={'depth': 2, 'hidden_size': 128, 'intermediate_size': 256, 'num_heads': 4,
        'patch_size': 16, 'temporal_patch_size': 2, 'spatial_merge_size': 2, 'out_hidden_size': 128,
        'num_position_embeddings': 16, 'deepstack_visual_indexes': [0, 1]},
        image_token_id=251, video_token_id=253, vision_start_token_id=250, vision_end_token_id=252,
        eos_token_id=2, pad_token_id=0)
    for value in [config, config.text_config, config.vision_config]: value._attn_implementation = 'eager'
    torch.manual_seed(20261007)
    model = Qwen3VLForConditionalGeneration(config).eval().to(device='cuda', dtype=torch.float16)
    assert all(parameter.device.type == 'cuda' for parameter in model.parameters())
    assert model.config.text_config._attn_implementation == 'eager'
    parameters = sum(p.numel() for p in model.parameters())
    with torch.inference_mode():
        text = torch.tensor([[1, 8, 9, 7]], device='cuda')
        output = model(input_ids=text, attention_mask=torch.ones_like(text))
        assert output.logits.shape == (1, 4, 256) and torch.isfinite(output.logits).all().item()
        generated = model.generate(input_ids=text, attention_mask=torch.ones_like(text), max_new_tokens=2, do_sample=False)
        assert 4 < generated.shape[1] <= 6
        report['tiny_Qwen_text_passed'] = True
        save()
        ids = torch.tensor([[1, 250, 251, 251, 251, 251, 252, 7]], device='cuda')
        pixels = torch.randn(16, 1536, device='cuda', dtype=torch.float16)
        grid = torch.tensor([[1, 4, 4]], dtype=torch.int64, device='cuda')
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixels, image_grid_thw=grid)
        assert output.logits.shape == (1, 8, 256) and torch.isfinite(output.logits).all().item()
        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixels,
                                   image_grid_thw=grid, max_new_tokens=2, do_sample=False)
        assert 8 < generated.shape[1] <= 10
        report['tiny_Qwen_multimodal_passed'] = True
    del model
    torch.cuda.empty_cache()
    return {'random_initialized_parameters': parameters, 'pretrained_weights_used': False,
            'tiny_text_and_multimodal_forward_and_bounded_greedy_generation': True,
            'semantic_or_answer_quality_test': False}


if __name__ == '__main__':
    qualification_lock = ROOT / 'state/gpu-reader-qualification.lock'
    descriptor = os.open(qualification_lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    atexit.register(qualification_lock.unlink)
    assert not REPORT.exists(), 'retain previous real qualification; audit any rerun separately'
    assert platform.machine() == 'aarch64' and sys.version_info[:3] == (3, 12, 14)
    assert Path(sys.prefix).resolve() == (ROOT / 'env').resolve()
    assert json.loads((ROOT / 'state/vision-completed.json').read_text())['status'] == 'completed'
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    sys.addaudithook(deny_network_during_no_weight_gates)
    save()
    try:
        import torch
        import torchvision
        import transformers
        assert torch.__version__.split('+')[0] == '2.3.1'
        # A GitHub source archive has no .git object database. Preserve upstream's
        # Unknown value; bind provenance to fixed official URL/archive/build wheel.
        assert torch.version.git_version in ['63d5e9221bedd1546b7d364b5ce4171547db12a9', 'Unknown']
        assert torchvision.__version__.split('+')[0] == '0.18.1' and transformers.__version__ == '4.57.6'
        assert torch.version.cuda == '11.4' and torch.cuda.is_available()
        torch.set_num_threads(2)
        torch.manual_seed(20261007)
        report['diagnostic_random_seed'] = 20261007
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        report['runtime'] = {'Python': platform.python_version(), 'torch': torch.__version__,
            'torch_runtime_git_version': torch.version.git_version, 'torchvision': torchvision.__version__,
            'Transformers': transformers.__version__, 'CUDA': torch.version.cuda,
            'cuDNN': torch.backends.cudnn.version(), 'GPU_name': torch.cuda.get_device_name(0),
            'GPU_capability': list(torch.cuda.get_device_capability(0)),
            'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'Pillow', 'safetensors', 'tokenizers']},
            'runtime_flash_and_mem_efficient_SDP_disabled': True}
        provenance = {}
        for label, repository in [('torch', 'https://github.com/pytorch/pytorch'), ('vision', 'https://github.com/pytorch/vision')]:
            source = next(row for row in PLAN['resources'] if row.get('repository') == repository)
            receipt = json.loads((ROOT / 'state' / (source['id'] + '-receipt.json')).read_text())
            assert file_sha256(ROOT / 'downloads' / source['filename']) == receipt['sha256']
            wheels = list((ROOT / 'sources' / label / 'dist').glob('*.whl'))
            assert len(wheels) == 1
            provenance[label] = {'fixed_official_source_commit': source['commit'],
                'source_archive_sha256': receipt['sha256'], 'compiled_wheel_sha256': file_sha256(wheels[0]),
                'compiled_wheel_bytes': wheels[0].stat().st_size}
        report['build_provenance'] = provenance
        step('actual_CUDA_operators', cuda_operators)
        step('native_processor_and_runtime', native_processor)
        step('random_initialized_tiny_Qwen_eager', tiny_qwen)
        assert all(report[key] for key in ['CUDA_operators_passed','tiny_Qwen_text_passed','tiny_Qwen_multimodal_passed','native_processor_and_runtime_requalified'])
        report.update(status='passed', completed_utc=datetime.now(timezone.utc).isoformat(),
                      CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save()
    except Exception as error:
        report.update(status='stopped', type=type(error).__name__,
                      reason=str(error).replace(str(Path.home()), '~'), completed_utc=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps(report), flush=True)
        raise SystemExit(3)
    print(json.dumps({'status': report['status'], 'gates_passed': True, 'pretrained_weights_loaded': False}), flush=True)
