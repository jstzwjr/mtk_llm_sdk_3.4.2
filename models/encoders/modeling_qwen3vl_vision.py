"""Define Qwen3-VL Vision model class.

Adapted from modeling_qwen2_5vl_vision.py with the following changes:
- MLP: SwiGLU (gate+up+down) -> Standard 2-layer (linear_fc1 + GELU + linear_fc2)
- Norm: RMSNorm (no bias) -> LayerNorm (with bias)
- Position encoding: RoPE -> Learned positional embedding
- Window attention: Removed (Qwen3-VL uses full attention only)
- PatchEmbed: Added bias support
"""

import math

import mtk_quantization
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ...utils import logger, utils
from ..modeling_base import BaseVisionEncoderChunk
from .modeling_qwen2_5vl_vision import precompute_qwen2_5vl_vision_rot_emb


class Conv2dInplaceConv3d(torch.nn.Module):
    """Replace Conv3d with two Conv2d for MTK compatibility."""

    def __init__(self, conv3d):
        super().__init__()
        weight = conv3d.weight  # [out, in, T=2, H, W]
        self.conv2d1 = nn.Conv2d(
            weight.shape[1], weight.shape[0], kernel_size=weight.shape[3], stride=weight.shape[3], bias=False
        )
        self.conv2d2 = nn.Conv2d(
            weight.shape[1], weight.shape[0], kernel_size=weight.shape[3], stride=weight.shape[3], bias=False
        )
        self.conv2d1.weight = nn.Parameter(weight[:, :, 0, :, :].contiguous())
        self.conv2d2.weight = nn.Parameter(weight[:, :, 1, :, :].contiguous())
        self.has_bias = conv3d.bias is not None
        if self.has_bias:
            self.bias = conv3d.bias

    def forward(self, x):
        # Keep as 4D to avoid NeuroPilot >4D rank error
        out = self.conv2d1(x[:, 0::2]) + self.conv2d2(x[:, 1::2])  # [batch, embed_dim, H, W]
        if self.has_bias:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class PatchEmbed(nn.Module):
    """Patch embedding for Qwen3-VL (with bias support)."""

    def __init__(self, config, patch_size=16, temporal_patch_size=2, in_channels=3, embed_dim=1024):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.use_conv2d_patch_embed = config.use_conv2d_patch_embed

        if self.use_conv2d_patch_embed:
            self.proj = Conv2dInplaceConv3d(
                nn.Conv3d(in_channels, embed_dim, kernel_size=[temporal_patch_size, patch_size, patch_size],
                          stride=[temporal_patch_size, patch_size, patch_size], bias=True)
            )
        else:
            self.proj = nn.Conv3d(
                in_channels, embed_dim, kernel_size=[temporal_patch_size, patch_size, patch_size],
                stride=[temporal_patch_size, patch_size, patch_size], bias=True
            )

    def forward(self, hidden_states, target_dtype=torch.float32):
        if self.use_conv2d_patch_embed:
            hidden_states = hidden_states.view(
                -1, self.in_channels * self.temporal_patch_size, self.patch_size, self.patch_size
            )
        else:
            hidden_states = hidden_states.view(
                -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
            )
        out = self.proj(hidden_states.to(target_dtype))
        if out.dim() == 5:
            return out.view(-1, self.embed_dim)
        else:
            # Conv2d output is 4D: [batch, embed_dim, H, W] -> [batch*H*W, embed_dim]
            return out.permute(0, 2, 3, 1).reshape(-1, self.embed_dim)


class Qwen3VLMLP(nn.Module):
    """Standard 2-layer MLP with GELU activation (not SwiGLU)."""

    def __init__(self, config, bias=True):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.embed_dim, config.intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.embed_dim, bias=bias)
        self.act_fn = nn.GELU(approximate='tanh')

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


def _qwen3vl_rotate_half(x):
    """Rotate the last dim by half (HF convention)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _qwen3vl_apply_rotary(q, k, packed_freqs):
    """Apply RoPE to vision Q/K aligned with HF Qwen3VL convention.

    Args:
        q, k: (1, seq, num_heads, head_dim)
        packed_freqs: (seq, 2, head_dim/2) packed [cos_half, sin_half] from
            precompute_qwen2_5vl_vision_rot_emb.

    Returns:
        Tuple of (q_rot, k_rot) with same shape & dtype as input.
    """
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos_half = packed_freqs[:, 0, :].float()  # (seq, head_dim/2)
    sin_half = packed_freqs[:, 1, :].float()
    cos = torch.cat([cos_half, cos_half], dim=-1)  # (seq, head_dim)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_rot = (q * cos) + (_qwen3vl_rotate_half(q) * sin)
    k_rot = (k * cos) + (_qwen3vl_rotate_half(k) * sin)
    return q_rot.to(orig_q_dtype), k_rot.to(orig_k_dtype)


class VisionAttention(nn.Module):
    """Vision attention (full attention only, no windowing)."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states, attn_mask=None, pos_embed=None):
        # hidden_states: [seq_len, dim] (2D, no batch)
        L, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(L, 3, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        q, k, v = qkv.unbind(0)  # each: [num_heads, seq_len, head_dim]

        # NEW: apply RoPE on Q/K (HF Qwen3VL aligns this every layer)
        if pos_embed is not None:
            # Reshape to (1, L, H, head_dim) for HF-style apply
            q4 = q.permute(1, 0, 2).unsqueeze(0).contiguous()
            k4 = k.permute(1, 0, 2).unsqueeze(0).contiguous()
            q4, k4 = _qwen3vl_apply_rotary(q4, k4, pos_embed)
            q = q4.squeeze(0).permute(1, 0, 2).contiguous()
            k = k4.squeeze(0).permute(1, 0, 2).contiguous()

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(1, 0, 2).reshape(L, -1)
        return self.proj(attn_output)


class Qwen3VLVisionBlock(nn.Module):
    """Vision block with LayerNorm (with bias) instead of RMSNorm."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.attn = VisionAttention(config, layer_idx)
        self.mlp = Qwen3VLMLP(config, bias=True)

    def forward(self, hidden_states, attn_mask=None, pos_embed=None):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), attn_mask=attn_mask, pos_embed=pos_embed,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class CumulativeSequenceLens(nn.Module):
    """Compute cumulative sequence lengths from grid_thw."""

    def forward(self, grid_thw):
        cu_seqlens = torch.zeros(1, dtype=torch.int32)
        for t, h, w in grid_thw:
            seq_len = int(t) * int(h) * int(w)
            cu_seqlens = torch.cat([cu_seqlens, cu_seqlens[-1:] + seq_len])
        return cu_seqlens


class Qwen3VLVisionModel(BaseVisionEncoderChunk):
    """Qwen3-VL Vision Encoder Model."""

    def __init__(self, config, lora=None, num_layers=1, first_layer_idx=0, chunk_idx=0, dtype=torch.float32, **kwargs):
        super().__init__(config, lora, num_layers, first_layer_idx, chunk_idx, dtype, **kwargs)
        self.config = config
        self.chunk_idx = chunk_idx
        self.first_layer_idx = first_layer_idx
        self.num_layers = num_layers
        self.dtype = dtype

        # Patch embedding (only in first chunk)
        if chunk_idx == 0:
            self.patch_embed = PatchEmbed(
                config,
                patch_size=config.patch_size,
                temporal_patch_size=config.temporal_patch_size,
                in_channels=config.in_channels,
                embed_dim=config.embed_dim,
            )
            # Learned positional embedding
            self.pos_embed = nn.Embedding(config.num_position_embeddings, config.embed_dim)

        # Vision blocks
        self.layers = nn.ModuleList([
            Qwen3VLVisionBlock(config, self.first_layer_idx + i) for i in range(self.num_layers)
        ])

        self._generate_default_state_dict_mapping()

    def _generate_default_state_dict_mapping(self):
        state_dict_mapping = {}

        if self.chunk_idx == 0:
            # Patch embed weights (with bias)
            if self.config.use_conv2d_patch_embed:
                state_dict_mapping.update({
                    'patch_embed_conv2d1_weight': {'patch_embed.proj.conv2d1.weight': 'visual.patch_embed.proj.weight'},
                    'patch_embed_conv2d2_weight': {'patch_embed.proj.conv2d2.weight': 'visual.patch_embed.proj.weight'},
                })
            else:
                state_dict_mapping['patch_embed_weight'] = {
                    'patch_embed.proj.weight': 'visual.patch_embed.proj.weight'
                }
            state_dict_mapping['patch_embed_bias'] = {
                'patch_embed.proj.bias': 'visual.patch_embed.proj.bias'
            }
            # Learned positional embedding
            state_dict_mapping['pos_embed_weight'] = {
                'pos_embed.weight': 'visual.pos_embed.weight'
            }

        for inner_idx, outer_idx in enumerate(
            range(self.first_layer_idx, self.first_layer_idx + self.num_layers)
        ):
            state_dict_mapping.update({
                f'{outer_idx}.attn.proj_bias': {
                    f'layers.{inner_idx}.attn.proj.bias': f'visual.blocks.{outer_idx}.attn.proj.bias'
                },
                f'{outer_idx}.attn.proj_weight': {
                    f'layers.{inner_idx}.attn.proj.weight': f'visual.blocks.{outer_idx}.attn.proj.weight'
                },
                f'{outer_idx}.attn.qkv_bias': {
                    f'layers.{inner_idx}.attn.qkv.bias': f'visual.blocks.{outer_idx}.attn.qkv.bias'
                },
                f'{outer_idx}.attn.qkv_weight': {
                    f'layers.{inner_idx}.attn.qkv.weight': f'visual.blocks.{outer_idx}.attn.qkv.weight'
                },
                # MLP: linear_fc1 / linear_fc2 (standard MLP, not SwiGLU)
                f'{outer_idx}.mlp.fc1_weight': {
                    f'layers.{inner_idx}.mlp.linear_fc1.weight': f'visual.blocks.{outer_idx}.mlp.linear_fc1.weight'
                },
                f'{outer_idx}.mlp.fc1_bias': {
                    f'layers.{inner_idx}.mlp.linear_fc1.bias': f'visual.blocks.{outer_idx}.mlp.linear_fc1.bias'
                },
                f'{outer_idx}.mlp.fc2_weight': {
                    f'layers.{inner_idx}.mlp.linear_fc2.weight': f'visual.blocks.{outer_idx}.mlp.linear_fc2.weight'
                },
                f'{outer_idx}.mlp.fc2_bias': {
                    f'layers.{inner_idx}.mlp.linear_fc2.bias': f'visual.blocks.{outer_idx}.mlp.linear_fc2.bias'
                },
                # LayerNorm (with bias)
                f'{outer_idx}.norm1_weight': {
                    f'layers.{inner_idx}.norm1.weight': f'visual.blocks.{outer_idx}.norm1.weight'
                },
                f'{outer_idx}.norm1_bias': {
                    f'layers.{inner_idx}.norm1.bias': f'visual.blocks.{outer_idx}.norm1.bias'
                },
                f'{outer_idx}.norm2_weight': {
                    f'layers.{inner_idx}.norm2.weight': f'visual.blocks.{outer_idx}.norm2.weight'
                },
                f'{outer_idx}.norm2_bias': {
                    f'layers.{inner_idx}.norm2.bias': f'visual.blocks.{outer_idx}.norm2.bias'
                },
            })

        self.state_dict_mapping = state_dict_mapping

    def load_weights(self, state_dict, state_dict_start_idx, quant_config=None):
        """Load model weights from state dict."""
        logger.debug('Enter Qwen3VLVisionModel load_weights')
        if state_dict is None:
            logger.error('state_dict cannot be None', err=ValueError)

        self.device_list = []
        self.prefixes = ['']
        weights_to_load = {}
        state_dict_keys = list(state_dict.keys())
        missing_keys = []

        # Detect prefix (model.visual. vs visual.)
        for k in state_dict_keys:
            if 'visual.patch_embed' in k:
                prefix = k.split('visual.')[0]
                if prefix and prefix not in self.prefixes:
                    self.prefixes.append(prefix)
                break

        # Handle conv2d patch embed
        if self.chunk_idx == 0 and self.config.use_conv2d_patch_embed:
            for pre in self.prefixes:
                key = f'{pre}visual.patch_embed.proj.weight'
                if key in state_dict_keys:
                    conv3dweight = state_dict.pop(key)
                    weights_to_load['patch_embed.proj.conv2d1.weight'] = conv3dweight[:, :, 0, :, :].contiguous().to(torch.float32)
                    weights_to_load['patch_embed.proj.conv2d2.weight'] = conv3dweight[:, :, 1, :, :].contiguous().to(torch.float32)
                    self.state_dict_mapping.pop('patch_embed_conv2d1_weight', None)
                    self.state_dict_mapping.pop('patch_embed_conv2d2_weight', None)
                    # Handle bias
                    bias_key = f'{pre}visual.patch_embed.proj.bias'
                    if bias_key in state_dict_keys:
                        weights_to_load['patch_embed.proj.bias'] = state_dict.pop(bias_key).to(torch.float32)
                        self.state_dict_mapping.pop('patch_embed_bias', None)
                    break

        for internal_key, mapping_dict in list(self.state_dict_mapping.items()):
            model_key = next(iter(mapping_dict))
            external_key = mapping_dict[model_key]
            dtype = torch.float32  # Keep everything in float32 for encoder

            found = False
            for pre in self.prefixes:
                key_to_test = pre + external_key
                if key_to_test in state_dict_keys:
                    weights_to_load[model_key] = state_dict.pop(key_to_test).to(dtype)
                    found = True
                    break

            if not found:
                missing_keys.append(external_key)

        if missing_keys:
            logger.warning(f'Missing {len(missing_keys)} encoder weights: {missing_keys[:5]}...')

        # Load weights into model
        load_result = self.load_state_dict(weights_to_load, strict=False)
        if load_result.unexpected_keys:
            logger.debug(f'Unexpected keys: {load_result.unexpected_keys}')

        # Set device placement (stay on CPU if jit_trace mode)
        import os
        num_gpu = torch.cuda.device_count()
        if num_gpu == 0 or getattr(self, 'jit_trace', False):
            self.device_list = ['cpu' for _ in range(self.num_layers)]
        else:
            gpu_id = os.getenv('LOCAL_RANK', 0)
            self.device_list = [f'cuda:{gpu_id}' for _ in range(self.num_layers)]

        # Move model to device
        target_device = self.device_list[0]
        for i, layer in enumerate(self.layers):
            layer.to(self.device_list[i])
        if self.chunk_idx == 0:
            self.patch_embed.to(target_device)
            self.pos_embed.to(target_device)

        return self, state_dict

    def _calculate_ptq_fixed_shape_batch(self):
        """Calculate fixed shape for PTQ from config."""
        from ..preprocessors.configuration_qwen2vl_vision import Qwen2VLPreprocessorConfig
        from ..preprocessors.preprocessor_qwen2vl_vision import Qwen2VLImageProcessor
        import numpy as np
        from PIL import Image

        preprocessor_config = Qwen2VLPreprocessorConfig(**self.config.preprocessor_config)
        processor = Qwen2VLImageProcessor(**preprocessor_config.get())
        image_hw = self.config.image_resolution
        if image_hw is None:
            image_hw = [448, 224]
        image = Image.fromarray(np.random.rand(image_hw[0], image_hw[1], 3).astype('uint8'))
        return_dict, _ = processor.preprocess([image])
        self.image_grid_thw = torch.tensor(return_dict['image_grid_thw'])
        # Pre-compute interpolated position embedding for the fixed shape (only chunk 0 has pos_embed)
        if self.chunk_idx == 0:
            # detach() to avoid "Cannot insert a Tensor that requires grad as a constant" during JIT trace
            self._precomputed_pos_embed = self._fast_pos_embed_interpolate(self.image_grid_thw).detach()

    def _fast_pos_embed_interpolate(self, grid_thw):
        """Bilinear interpolation of learned position embeddings (from HF Qwen3-VL).

        Interpolates the 48x48 position embedding grid to arbitrary h x w grid,
        then rearranges according to spatial_merge_size.
        """
        num_grid_per_side = int(self.config.num_position_embeddings ** 0.5)  # 48
        merge_size = getattr(self.config, 'spatial_merge_size', 2)
        device = self.pos_embed.weight.device

        grid_thw_list = grid_thw.tolist()
        grid_ts = [int(row[0]) for row in grid_thw_list]
        grid_hs = [int(row[1]) for row in grid_thw_list]
        grid_ws = [int(row[2]) for row in grid_thw_list]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            h, w = int(h), int(w)
            h_idxs = torch.linspace(0, num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * num_grid_per_side
            base_h_ceil = h_idxs_ceil * num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def _calculate_ptq_attn_mask(self):
        """Pre-compute attention mask for PTQ/JIT trace."""
        batch_size = int((self.image_grid_thw[0][1] * self.image_grid_thw[0][2]).item())
        cu_seqlens = CumulativeSequenceLens()(self.image_grid_thw)
        attn_mask = torch.full([1, batch_size, batch_size], self.config.mask_value, dtype=torch.float32)
        for i in range(1, len(cu_seqlens)):
            attn_mask[..., cu_seqlens[i - 1]:cu_seqlens[i], cu_seqlens[i - 1]:cu_seqlens[i]] = 0
        self._vision_attention_mask = attn_mask

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass. hidden_states: [seq_len, features] (2D, no batch dim)."""
        # Get or use pre-stored attention mask
        attn_mask = kwargs.get('qwen3vl_attn_mask')
        if attn_mask is None:
            attn_mask = getattr(self, '_vision_attention_mask', None)

        target_device = self.device_list[0] if self.device_list else 'cpu'
        hidden_states = hidden_states.to(target_device)

        if self.first_layer_idx == 0:
            # Patch embedding: [seq_len, flatten_patch] -> [seq_len, embed_dim]
            hidden_states = self.patch_embed(hidden_states, target_dtype=self.dtype)

            # Add learned positional embedding (interpolated for arbitrary resolution)
            precomputed = getattr(self, '_precomputed_pos_embed', None)
            if precomputed is not None:
                hidden_states = hidden_states + precomputed.to(device=hidden_states.device, dtype=hidden_states.dtype)
            else:
                # Fallback: simple lookup (only works when seq_len <= num_position_embeddings)
                seq_len = hidden_states.shape[0]
                if seq_len <= self.config.num_position_embeddings:
                    pos_ids = torch.arange(seq_len, device=hidden_states.device)
                    hidden_states = hidden_states + self.pos_embed(pos_ids)

        if attn_mask is not None:
            attn_mask = attn_mask.to(device=target_device)

        # NEW: rotary pos emb — read precomputed freqs (set by pipeline) or compute on demand
        rot_freqs = getattr(self, '_vision_rot_emb', None)
        if rot_freqs is None and getattr(self, 'image_grid_thw', None) is not None:
            rot_freqs = precompute_qwen2_5vl_vision_rot_emb(self.image_grid_thw, self.config)
        if rot_freqs is not None:
            rot_freqs = rot_freqs.to(device=target_device)

        # NEW: deepstack — collect intermediate hidden_states at configured indexes
        deepstack_indexes = list(getattr(self.config, 'deepstack_visual_indexes', []) or [])
        captured_deepstack = []

        # Forward through vision blocks: [seq_len, dim] -> [seq_len, dim]
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states.to(self.device_list[idx]),
                attn_mask=attn_mask.to(self.device_list[idx]) if attn_mask is not None else None,
                pos_embed=rot_freqs.to(self.device_list[idx]) if rot_freqs is not None else None,
            )
            # Capture this layer's output if it is a deepstack tap point
            global_idx = self.first_layer_idx + idx
            if global_idx in deepstack_indexes:
                # NEW (Patch A2): .clone() 让 jit_trace 给每个返回张量独立 op 名，避免 Duplicated output names
                captured_deepstack.append(hidden_states.clone())

        # Side-channel: pipeline.forward_encoder reads this attribute to aggregate across chunks
        # (kept for backward-compat with pipeline path that does NOT consume tuple return)
        self._last_deepstack_intermediates = captured_deepstack

        # NEW (Patch A1): if any deepstack intermediates captured, return them as part of tuple,
        # so jit_trace exposes them as graph outputs (PTQ-traceable, can flow into DLA out_ports).
        if captured_deepstack:
            return (hidden_states.clone(), *captured_deepstack)
        return hidden_states

    def get_jit_trace_inputs(self):
        """Get dummy inputs for JIT tracing."""
        self._calculate_ptq_fixed_shape_batch()
        self._calculate_ptq_attn_mask()
        fixed_batch_size = int((self.image_grid_thw[0][1] * self.image_grid_thw[0][2]).item())
        flatten_patch_size = (
            self.config.patch_size * self.config.patch_size * self.config.in_channels * self.config.temporal_patch_size
        )
        if self.chunk_idx == 0:
            temp_input = torch.randn(fixed_batch_size, flatten_patch_size, device='cpu', dtype=torch.float32)
            self.chunk0_input_shape = temp_input.shape
            return temp_input
        feature_size = int(self.config.embed_dim)
        return torch.randn(fixed_batch_size, feature_size, device='cpu', dtype=torch.float32)

    def get_ptq_inputs(self, args=None, **kwargs):
        """Get inputs for post-training quantization."""
        import numpy as np

        if hasattr(self, 'image_grid_thw') and self.image_grid_thw is not None:
            fixed_batch_size = int((self.image_grid_thw[0][1] * self.image_grid_thw[0][2]).item())
        else:
            h, w = 448, 224
            fixed_batch_size = (h // self.config.patch_size) * (w // self.config.patch_size)

        flatten_patch_size = (
            self.config.patch_size * self.config.patch_size * self.config.in_channels * self.config.temporal_patch_size
        )

        if self.chunk_idx == 0:
            input_shapes = [[fixed_batch_size, flatten_patch_size]]
        else:
            feature_size = int(self.config.embed_dim)
            input_shapes = [[fixed_batch_size, feature_size]]

        if hasattr(self, 'chunk0_input_shape'):
            input_shapes = [self.chunk0_input_shape] if self.chunk_idx == 0 else input_shapes

        input_value_ranges = [None]

        if args is not None and hasattr(args, 'calibration_dataset') and args.calibration_dataset != 'fake':
            import os as _os
            from ...utils import utils as _utils
            def calib_data_gen():
                for f in _utils.get_sorted_path_list(
                    _os.path.join(args.calibration_dataset, 'encoder', f'chunk_{self.chunk_idx}'), '.npz', sep='-'
                ):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]
        else:
            def calib_data_gen():
                for _ in range(10):
                    if self.chunk_idx == 0:
                        yield [np.random.rand(fixed_batch_size, flatten_patch_size).astype(np.float32)]
                    else:
                        feature_size = int(self.config.embed_dim)
                        yield [np.random.rand(fixed_batch_size, feature_size).astype(np.float32)]

        def eval_data_gen():
            return calib_data_gen()

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen
