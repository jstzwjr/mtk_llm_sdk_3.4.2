# Copyright (C) 2024 MediaTek Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# ==================================================================================================
"""Define constants for mtk_llm_sdk."""

import numpy as np

SUPPORTED_PREPROCESSORS = [
    'clip',
    'siglip',
    'intern_vit_6b',
    'qwen2_vl_vision',
    'phi3v',
    'minicpmv_navit_siglip',
    'whisper',
    'qwen2_vl',
    'gecko2_vision',
    'phi4o_img',
    'intern_vit_6b_navit_rope',
]
SUPPORTED_ENCODERS = [
    'clip',
    'siglip',
    'intern_vit_6b',
    'qwen2_vl_vision',
    'minicpmv_navit_siglip',
    'phi3_vision_emb',
    'whisper',
    'gecko2_vision',
    'phi4o_navit_siglip',
    'intern_vit_6b_navit_rope',
    'qwen2_5_vl_vision',
    'qwen2_audio_encoder',
]
SUPPORTED_PROJECTORS = [
    'mlp_gelu',
    'mlp_downsample',
    'ldpnetv2',
    'internvl2',
    'qwen2_vl',
    'phi3v',
    'gecko2_vision',
    'phi4o',
    'andesvl',
    'qwen2_5_vl',
    'qwen2_audio_encoder',
    'minicpmv',
]
SUPPORTED_LLMS = [
    'llama',
    'baichuan',
    'qwen',
    'qwen1.5',
    'qwen2',
    'qwen3',
    'milm',
    'phi3',
    'hunyuan',
    'internlm2',
    'minicpm',
    'gemma',
    'gemma2',
    'gemma3',
    'whisper_decoder',
    'gecko',
    'gecko2',
    'phi4',
]
SUPPORTED_CUSTOM_TAILS = ['medusa', 'eagle']
SUPPORTED_MODELS = (
    SUPPORTED_PREPROCESSORS + SUPPORTED_ENCODERS + SUPPORTED_PROJECTORS + SUPPORTED_LLMS + SUPPORTED_CUSTOM_TAILS
)

SUPPORTED_TOKENIZERS = [
    'default',
    'baichuan',
    'gpt2',
    'gpt2_fast',
    'qwen',
    'qwen2',
    'qwen2_fast',
    'llama',
    'llama_fast',
    'gemma',
    'gemma_fast',
    'pretrained_fast',
    'sentencepiece',
    'hunyuan',
    'internlm2',
    'internlm2_fast',
    'milm',
    'whisper',
    'whisper_fast',
]

TAIL_TYPES = ['tail', 'medusa', 'eagle']

DEFAULT_AUDIO_TOKEN = '<audio>'  # FIXME: placeholder
DEFAULT_AUDIO_TOKEN_ID = -1  # FIXME: placeholder
DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_TOKEN_ID = -200
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
VALID_SIZE_DICT_KEYS = (
    {'height', 'width'},
    {'shortest_edge'},
    {'shortest_edge', 'longest_edge'},
    {'longest_edge'},
)

# For InternVL2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# For Qwen2VL
QWEN2VL_IMAGE_TOKEN = '<|vision_start|><|image_pad|><|vision_end|>'

# For Phi3-V
PHI3V_HD_IMAGE_SIZE = [[1344, 1344]]
PHI3V_DEFAULT_IMAGE_TOKEN = '<|image_1|>'

# For minicpmv
MINICPMV_IMAGE_TOKEN = '(<image>./</image>)'

# For InternVL2
INTERNVL2_IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
INTERNVL2_IMG_START_TOKEN = '<img>'
INTERNVL2_IMG_END_TOKEN = '</img>'

SUPPORTED_COMBINED_CONFIGS = [
    'llava',  # clip + vicuna7b
    'siglip_downsample_llava',
    'tiny_llava',
    'mobilevlm',
    'llava-next',
    'paligemma',
    'vila',
    'llava-ov',
    'internvl2-1b',
    'internvl2-2b',
    'internvl2-4b',
    'internvl2-8b',
    'qwen2_vl',
    'phi3v',
    'whisper',
    'minicpmv',
]

CONVERTER = 'converter'
MLKITS = 'mlkits'
SUPPORTED_BACKENDS = [CONVERTER, MLKITS]

FLOAT_PIPELINE_TASKS = {
    'make_calibration',
    'ptq',
    'inference',
    'qalft',
    'evaluate',
    'test_tokenizer',
    'find_overture',
    'export_lora',
    'fuse_embed_ln',
    'hotplug',
}
QUANTIZED_PIPELINE_TASKS = {'inference', 'export_lora', 'evaluate'}
ALL_SDK_TASKS = FLOAT_PIPELINE_TASKS.union(QUANTIZED_PIPELINE_TASKS)

PIPELINE_CORE_MODULES = [
    'preprocessor',
    'encoder',
    'projector',
    'llm',
    'tail',
]
PIPELINE_HOOK_MODULES = [
    'format_text',
    'pre_preprocessor_hook',
    'pre_tokenizer_hook',
    'tokenizer_func_hook',
    'pre_encoder_hook',
    'pre_projector_hook',
    'pre_getembed_hook',
    'get_embeds',
    'pre_llm_hook',
    'pre_tail_hook',
    'logits_processor',
    'stop_criteria',
]
ALL_PIPELINE_MODULES = PIPELINE_CORE_MODULES + PIPELINE_HOOK_MODULES

SUPPORTED_HOOKS = [
    'passthrough',
    'patch_select',
    'numpy_to_torch',
    'torch_to_numpy',
    'internvl2_pixel_shuffle',
    'qwen2vl_prellm',
    'qwen2vl_preencoder',
    'pad_internvl2',
    'phi3v_preprojector',
    'pad_phi3v',
    'phi4o_img_preprojector',
    'andesvl',
    'andesvl_preprojector',
    'andesvl_preencoder',
    'qwen2_5vl_preencoder',
    'bita_pre_getembed',
]
SUPPORTED_TOKENIZER_FUNCS = ['default', 'internvl2', 'phi3v', 'phi4o']
SUPPORTED_GENMODS = [
    'passthrough',
    'whisper',
]

SUPPORTED_GETEMBEDS = [
    'text_only',
    'llava',
    'internvl2',
    'qwen2_vl',
    'phi3v',
    'gecko2',
    'andesvl',
    'qwen2_5_vl',
    'text_with_mm',
    'minicpmv',
]
SUPPORTED_FORMATTEXT = ['passthrough', 'llava', 'internvl2', 'qwen2_vl', 'phi3v', 'phi4o', 'andesvl', 'minicpmv']

SUPPORTED_PRECISION_TARGETS = [
    'attention',
    'mlp',
]

SUPPORTED_METRICS = ['ppl', 'perplexity', 'bleu', 'logits']

INBUILT_DATASETS = [
    'wikitext',
    'c4',
]

EVICTION_INDEX_MAX_SCORE = 1e8
EVICTION_EVALUATION_MAX_NEW_TOKEN = 20

NP_COMPAT_VERIONS = [7, 8, 9]

DTYPE_ABSMAX_MAP = {
    np.uint8: 2**8 - 1,
    np.int8: 2**7 - 1,
    np.uint16: 2**16 - 1,
    np.int16: 2**15 - 1,
}

DEFAULT_JIT_TRACE_CACHE_SIZE = 128
DEFAULT_JIT_TRACE_NUM_TOKEN = 128
