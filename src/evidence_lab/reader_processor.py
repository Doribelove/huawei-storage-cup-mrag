"""Pinned native vision preprocessing and live tensor bindings. No model loading."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
from weakref import WeakKeyDictionary

from .reader_context import (TOKENIZER_FILES, PinnedReaderTokenizer, ReaderBudget,
    VisualReaderContext, visual_reader_context_verified, native_processor_messages)

PROCESSOR_FILES = {**TOKENIZER_FILES,
    'preprocessor_config.json': (390, '2ea84a437d448ff71b08df68fdd949d5cc4ebb64'),
    'video_preprocessor_config.json': (385, '3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe')}
RUNTIME_PINS = {'torch':'2.10.0+cpu', 'torchvision':'0.25.0+cpu',
    'transformers':'4.57.6', 'tokenizers':'0.22.2', 'numpy':'2.3.5', 'pillow':'12.3.0'}
RUNTIME_PROFILES = {'lenovo-cpu-v1':RUNTIME_PINS,
    'agx-r35-5-qualified-r2':{'torch':'2.3.1','torchvision':'0.18.1',
        'transformers':'4.57.6','tokenizers':'0.22.2','numpy':'1.26.4','pillow':'11.3.0'}}


class VisualProcessorUnavailable(RuntimeError): pass


@dataclass(frozen=True)
class VisualPixelPolicy:
    min_pixels: int = 65536
    max_pixels: int = 524288

    def validate(self):
        if (type(self.min_pixels) is not int or type(self.max_pixels) is not int
                or not 65536 <= self.min_pixels <= self.max_pixels <= 16777216):
            raise ValueError('Use bounded integer pixel limits within the pinned configuration')


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _verify_files(directory):
    files = []
    for name in ('processor_config.json', 'added_tokens.json', 'special_tokens_map.json',
                 'tokenizer.model', 'chat_template.jinja', 'chat_templates'):
        if (directory/name).exists(): raise VisualProcessorUnavailable('Unverified processor/tokenizer sidecar')
    for name,(size,blob) in PROCESSOR_FILES.items():
        path = directory/name
        raw = path.read_bytes()
        if len(raw) != size or hashlib.sha1(b'blob '+str(size).encode()+b'\0'+raw).hexdigest() != blob:
            raise VisualProcessorUnavailable('Pinned processor file mismatch: '+name)
        files.append((str(path),hashlib.sha256(raw).hexdigest()))
    return tuple(files)


def _processor_signature(processor):
    image = processor.image_processor
    return _json_hash({'classes':[type(processor).__name__,type(image).__name__,
        type(processor.video_processor).__name__,type(processor.tokenizer).__name__],
        'size':dict(image.size),'min_pixels':image.min_pixels,'max_pixels':image.max_pixels,'resample':int(image.resample),
        'image_options':{name:getattr(image,name) for name in ('patch_size','temporal_patch_size','merge_size',
            'do_resize','do_convert_rgb','do_rescale','rescale_factor','do_normalize','image_mean','image_std')},
        'template':processor.chat_template,
        'tokenizer_sha256':hashlib.sha256(processor.tokenizer.backend_tokenizer.to_str().encode()).hexdigest()})


@dataclass(frozen=True)
class _LoadedProcessor:
    backend: object
    image: object
    video: object
    tokenizer: object
    signature: str
    files: tuple
    policy: VisualPixelPolicy
    context_tokenizer: object
    methods: tuple
    runtime_profile: str
    runtime_pins: tuple


_LOADED_PROCESSORS = WeakKeyDictionary()


def _verified_processor(value):
    if type(value) is not PinnedVisualProcessor: return None
    state = _LOADED_PROCESSORS.get(value)
    if state is None or getattr(value,'_backend',None) is not state.backend: return None
    from .safe_data import sha256_file
    try:
        if (state.backend.image_processor is not state.image or state.backend.video_processor is not state.video
                or state.backend.tokenizer is not state.tokenizer
                or getattr(value,'_tokenizer',None) is not state.context_tokenizer
                or any(getattr(getattr(obj,name),'__func__',None) is not function for obj,name,function in state.methods)
                or _processor_signature(state.backend) != state.signature
                or _verify_files(Path(state.files[0][0]).parent) != state.files
                or any(sha256_file(Path(p)) != digest for p,digest in state.files)):
            return None
        return state
    except (OSError, ValueError, TypeError, AttributeError, VisualProcessorUnavailable):
        return None


class PinnedVisualProcessor:
    """Only native AutoProcessor, fixed local JSONs and explicit synchronized policy."""
    def __init__(self, directory: Path, *, policy: VisualPixelPolicy = VisualPixelPolicy(),
                 runtime_profile: str = 'lenovo-cpu-v1'):
        if type(self) is not PinnedVisualProcessor or type(policy) is not VisualPixelPolicy:
            raise VisualProcessorUnavailable('Use the exact controlled native processor construction')
        policy.validate()
        if runtime_profile not in RUNTIME_PROFILES:
            raise VisualProcessorUnavailable('Unknown native processor runtime profile')
        pins = RUNTIME_PROFILES[runtime_profile]
        for name,version in pins.items():
            try: actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as exc:
                raise VisualProcessorUnavailable('Native processor dependency is absent: '+name) from exc
            if actual != version:
                raise VisualProcessorUnavailable('Native processor runtime mismatch: '+name)
        directory = Path(directory).resolve()
        files = _verify_files(directory)
        from transformers import (AutoProcessor, Qwen3VLProcessor, Qwen2TokenizerFast,
                                  Qwen2VLImageProcessorFast, Qwen3VLVideoProcessor)
        backend = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
            size={'shortest_edge':policy.min_pixels,'longest_edge':policy.max_pixels},
            min_pixels=policy.min_pixels,max_pixels=policy.max_pixels)
        if (type(backend) is not Qwen3VLProcessor or type(backend.image_processor) is not Qwen2VLImageProcessorFast
                or type(backend.video_processor) is not Qwen3VLVideoProcessor or type(backend.tokenizer) is not Qwen2TokenizerFast):
            raise VisualProcessorUnavailable('Unexpected native processor components')
        image = backend.image_processor
        if (dict(image.size) != {'shortest_edge':policy.min_pixels,'longest_edge':policy.max_pixels}
                or image.min_pixels != policy.min_pixels or image.max_pixels != policy.max_pixels
                or (image.patch_size,image.temporal_patch_size,image.merge_size) != (16,2,2)
                or image.image_mean != [.5,.5,.5] or image.image_std != [.5,.5,.5]
                or not all((image.do_resize,image.do_convert_rgb,image.do_rescale,image.do_normalize))):
            raise VisualProcessorUnavailable('Actual size/min/max/patch/normalization policy mismatch')
        self._backend = backend
        self._tokenizer = PinnedReaderTokenizer(directory)
        methods = []
        for obj,cls,names in ((backend,Qwen3VLProcessor,('apply_chat_template','__call__','decode')),
                              (image,Qwen2VLImageProcessorFast,('preprocess','__call__','_preprocess'))):
            for name in names:
                function = getattr(cls,name)
                if getattr(getattr(obj,name),'__func__',None) is not function:
                    raise VisualProcessorUnavailable('Native processing method was substituted')
                methods.append((obj,name,function))
        _LOADED_PROCESSORS[self] = _LoadedProcessor(backend,image,backend.video_processor,backend.tokenizer,
            _processor_signature(backend),files,policy,self._tokenizer,tuple(methods),runtime_profile,tuple(sorted(pins.items())))

    @property
    def identity(self):
        state = _verified_processor(self)
        if state is None: raise VisualProcessorUnavailable('Native processor live state is unverified or changed')
        return {'processor_class':type(state.backend).__name__,'image_class':type(state.image).__name__,
            'effective_size':dict(state.image.size),'min_pixels':state.image.min_pixels,'max_pixels':state.image.max_pixels,
            'patch_size':16,'temporal_patch_size':2,'merge_size':2,
            'runtime_profile':state.runtime_profile,'runtime_pins':dict(state.runtime_pins),'local_files_only':True,'trust_remote_code':False,
            'weights_loaded':False,'native_processor_verified':True,
            'file_sha256':dict(state.files)}


def _validate_native_batch(batch, placeholder_ids, *, image_token_id, image_count, policy,
                           budget, max_new_tokens):
    """Tensor contract mechanics, also exercised with explicitly synthetic fixtures."""
    import torch
    allowed = {'input_ids','attention_mask','pixel_values','image_grid_thw','mm_token_type_ids'}
    if not {'input_ids','attention_mask','pixel_values','image_grid_thw'} <= set(batch) <= allowed:
        raise VisualProcessorUnavailable('Missing/unknown native image tensor fields')
    if any(type(value) is not torch.Tensor or value.device.type != 'cpu' for value in batch.values()):
        raise VisualProcessorUnavailable('Use native CPU tensors from this preprocessing stage')
    ids, mask, pixels, grid = (batch[key] for key in ('input_ids','attention_mask','pixel_values','image_grid_thw'))
    if ids.dtype != torch.int64 or ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 65536:
        raise VisualProcessorUnavailable('Invalid single-prompt native input IDs')
    if mask.dtype != torch.int64 or mask.shape != ids.shape or not torch.all(mask == 1).item():
        raise VisualProcessorUnavailable('Unexpected padding/truncation or attention mask')
    if grid.dtype != torch.int64 or tuple(grid.shape) != (image_count,3):
        raise VisualProcessorUnavailable('Image grid count/dtype mismatch')
    grids = grid.tolist(); counts = []; expected_rows = 0
    for temporal,height,width in grids:
        if temporal != 1 or height <= 0 or width <= 0 or height % 2 or width % 2:
            raise VisualProcessorUnavailable('Invalid static-image grid or merge divisibility')
        area = height*width*16**2
        if not policy.min_pixels <= area <= policy.max_pixels:
            raise VisualProcessorUnavailable('Actual resized image area violates effective size/min/max')
        expected_rows += temporal*height*width
        counts.append(temporal*height*width//4)
    if (pixels.dtype != torch.float32 or tuple(pixels.shape) != (expected_rows,1536)
            or pixels.numel()*pixels.element_size() > 96*1024**2 or not torch.isfinite(pixels).all().item()):
        raise VisualProcessorUnavailable('Native pixel tensor shape/bytes/finite-value check failed')
    if 'mm_token_type_ids' in batch and (batch['mm_token_type_ids'].dtype != torch.int64
            or batch['mm_token_type_ids'].shape != ids.shape):
        raise VisualProcessorUnavailable('Unexpected multimodal type IDs')
    expected = []; index = 0
    for token in placeholder_ids:
        if token == image_token_id:
            if index >= image_count: raise VisualProcessorUnavailable('Too many image placeholders')
            expected.extend([image_token_id]*counts[index]); index += 1
        else: expected.append(token)
    if index != image_count or expected != ids[0].tolist():
        raise VisualProcessorUnavailable('Full expanded prompt differs from original placeholder sequence and actual grids')
    expanded = ids.shape[1]
    passed = expanded <= budget.max_prompt_tokens and expanded+max_new_tokens <= budget.max_total_tokens
    return {'status':'native_image_tokens_verified' if passed else 'blocked_expanded_prompt_budget',
        'token_budget_verified':passed,'tensor_contract_validated':True,'image_grid_thw':grids,
        'image_tokens_by_image':counts,'pixel_tensor_shape':list(pixels.shape),
        'pixel_tensor_bytes':pixels.numel()*pixels.element_size(),'pixel_values_finite':True,
        'expanded_prompt_tokens':expanded,'placeholder_prompt_tokens':len(placeholder_ids),
        'full_expanded_token_ids_equal':True,'max_prompt_tokens':budget.max_prompt_tokens,
        'max_total_tokens':budget.max_total_tokens,'max_new_tokens_reserved':max_new_tokens,
        'expanded_prompt_plus_reserve':expanded+max_new_tokens}


def _batch_hash(batch):
    import torch
    digest = hashlib.sha256()
    for name in sorted(batch):
        tensor = batch[name]
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(eq=False)
class NativeVisualInputs:
    prepared: VisualReaderContext
    processor: PinnedVisualProcessor
    tensors: dict
    report: dict


@dataclass(frozen=True)
class _BoundInputs:
    prepared: object
    processor: object
    tensor_hash: str
    report_hash: str


_NATIVE_INPUTS = WeakKeyDictionary()


def native_visual_inputs_verified(value):
    if type(value) is not NativeVisualInputs: return False
    state = _NATIVE_INPUTS.get(value)
    if state is None: return False
    try:
        return (value.prepared is state.prepared and value.processor is state.processor
            and visual_reader_context_verified(value.prepared) and _verified_processor(value.processor) is not None
            and _json_hash(value.report) == state.report_hash and _batch_hash(value.tensors) == state.tensor_hash
            and value.report['budget_validation']['token_budget_verified'] is True)
    except (ValueError,TypeError,RuntimeError,AttributeError,KeyError):
        return False


def prepare_native_visual_inputs(prepared, processor, *, budget=ReaderBudget(), max_new_tokens=256):
    """Real full image preprocessing. A blocked budget is reported without a ready packet."""
    from PIL import Image
    from .safe_data import sha256_file
    if not visual_reader_context_verified(prepared):
        raise VisualProcessorUnavailable('Visual source packet is unregistered, changed or stale')
    state = _verified_processor(processor)
    if state is None: raise VisualProcessorUnavailable('Native processor is unregistered, changed or unavailable')
    if type(budget) is not ReaderBudget: raise ValueError('Use controlled ReaderBudget')
    budget.validate()
    if type(max_new_tokens) is not int or not 32 <= max_new_tokens <= 512:
        raise ValueError('Native generation reserve must be 32..512 tokens')
    if (budget.max_prompt_tokens > prepared.report['budget_validation']['max_prompt_tokens']
            or budget.max_total_tokens > prepared.report['budget_validation']['max_total_tokens']):
        raise ValueError('Native preprocessing cannot loosen original source budgets')
    catalog = list(prepared.citation_catalog.values())
    if not 1 <= len(catalog) <= 4: raise ValueError('Native image count exceeded')
    for record in catalog:
        if sha256_file(Path(record['local_image_path'])) != record['image']['sha256']:
            raise VisualProcessorUnavailable('Source image bytes changed')
        with Image.open(record['local_image_path']) as image:
            if image.mode != 'RGB' or list(image.size) != record['image']['size_pixels']:
                raise VisualProcessorUnavailable('Source image dimensions/mode changed')
            image.load()
            if hashlib.sha256(image.tobytes()).hexdigest() != record['image']['rgb_pixels_sha256']:
                raise VisualProcessorUnavailable('Source image RGB pixels changed')
    encoded = PinnedReaderTokenizer.encode_prompt(processor._tokenizer,prepared.messages)
    rendered = state.backend.apply_chat_template(prepared.messages,add_generation_prompt=True,tokenize=False)
    if hashlib.sha256(rendered.encode()).hexdigest() != encoded.rendered_sha256:
        raise VisualProcessorUnavailable('Native and pinned templates differ')
    native_messages = native_processor_messages(prepared.messages)
    native_rendered = state.backend.apply_chat_template(native_messages,add_generation_prompt=True,tokenize=False)
    if native_rendered != rendered:
        raise VisualProcessorUnavailable('Typed native content changed full rendered prompt')
    batch = dict(state.backend.apply_chat_template(native_messages,add_generation_prompt=True,
        tokenize=True,return_dict=True,return_tensors='pt',truncation=False))
    image_id = state.backend.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    validation = _validate_native_batch(batch,encoded.token_ids,image_token_id=image_id,
        image_count=len(catalog),policy=state.policy,budget=budget,max_new_tokens=max_new_tokens)
    validation['actual_native_processor_run'] = True
    resized = []
    for record,(_,height,width) in zip(catalog,validation['image_grid_thw']):
        original_width,original_height = record['image']['size_pixels']
        resized.append({'image_id':record['image_id'],'original_image_size_pixels':record['image']['size_pixels'],
            'actual_processor_resized_image_size_pixels':[width*16,height*16],
            'original_to_processor_pixel_matrix':[width*16/original_width,0,0,height*16/original_height,0,0],
            'meaning':'Continuous resize coordinates; interpolation/semantic readability/localization unverified'})
    report = {'processor_identity':processor.identity,'budget_validation':validation,
        'source_image_order':[record['image_id'] for record in catalog],
        'actual_image_resizing':resized,'original_source_images_changed':False,
        'model_loaded':False,'model_generate_called':False,'generation_authorized':False,'supported_answer':False}
    if not visual_reader_context_verified(prepared) or _verified_processor(processor) is None:
        raise VisualProcessorUnavailable('Source or native processor changed during preprocessing')
    value = NativeVisualInputs(prepared,processor,batch,report)
    if validation['token_budget_verified']:
        _NATIVE_INPUTS[value] = _BoundInputs(prepared,processor,_batch_hash(batch),_json_hash(report))
    return value
