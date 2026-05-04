"""Qwen3-VL PatchMerger Projector with DeepStack support."""

import torch
import torch.nn as nn

from ...utils import logger
from ..modeling_base import BaseProjector


class Qwen3VLPatchMerger(BaseProjector):
    """Qwen3-VL merger: LayerNorm + Linear + GELU + Linear (plus deepstack mergers)."""

    def __init__(self, config, **kwargs):
        # Must create deepstack_merger_list before super().__init__
        # because base class calls _generate_default_state_dict_mapping
        self._num_deepstack = len(config.deepstack_visual_indexes) if hasattr(config, 'deepstack_visual_indexes') else 0
        super().__init__(config, **kwargs)
        embed_dim = config.embed_dim
        dim = config.dim  # LLM hidden_size
        spatial_merge_size = config.spatial_merge_size
        hidden_size = embed_dim * (spatial_merge_size ** 2)

        # Main merger
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.act = nn.GELU(approximate='tanh')
        self.linear_fc2 = nn.Linear(hidden_size, dim, bias=True)

        self.hidden_size = hidden_size
        self.spatial_merge_size = spatial_merge_size

        # DeepStack mergers
        self.deepstack_merger_list = nn.ModuleList()
        for _ in range(self._num_deepstack):
            self.deepstack_merger_list.append(nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_size, eps=1e-6),
                'linear_fc1': nn.Linear(hidden_size, hidden_size, bias=True),
                'act': nn.GELU(approximate='tanh'),
                'linear_fc2': nn.Linear(hidden_size, dim, bias=True),
            }))

        self._generate_default_state_dict_mapping()

    def _generate_default_state_dict_mapping(self):
        mapping = {
            'merger_norm_weight': {'norm.weight': 'visual.merger.norm.weight'},
            'merger_norm_bias': {'norm.bias': 'visual.merger.norm.bias'},
            'merger_fc1_weight': {'linear_fc1.weight': 'visual.merger.linear_fc1.weight'},
            'merger_fc1_bias': {'linear_fc1.bias': 'visual.merger.linear_fc1.bias'},
            'merger_fc2_weight': {'linear_fc2.weight': 'visual.merger.linear_fc2.weight'},
            'merger_fc2_bias': {'linear_fc2.bias': 'visual.merger.linear_fc2.bias'},
        }

        for i in range(self._num_deepstack):
            prefix = f'deepstack_merger_list.{i}'
            for name in ['norm.weight', 'norm.bias', 'linear_fc1.weight', 'linear_fc1.bias',
                         'linear_fc2.weight', 'linear_fc2.bias']:
                safe_name = name.replace('.', '_')
                mapping[f'{prefix}_{safe_name}'] = {
                    f'deepstack_merger_list.{i}.{name}': f'visual.deepstack_merger_list.{i}.{name}'
                }

        self.state_dict_mapping = mapping

    def load_weights(self, state_dict, **kwargs):
        logger.info('Enter Qwen3VLPatchMerger load_weights')
        logger.info(f'  state_dict has {len(state_dict)} keys')
        logger.info(f'  deepstack keys in sd: {[k for k in state_dict if "deepstack" in k][:5]}')
        logger.info(f'  merger keys in sd: {[k for k in state_dict if "merger" in k and "deepstack" not in k][:5]}')
        logger.info(f'  mapping has {len(self.state_dict_mapping)} entries')
        logger.info(f'  _num_deepstack: {self._num_deepstack}')
        for ik, md in self.state_dict_mapping.items():
            ek = next(iter(md.values()))
            logger.info(f'  mapping: {ik} -> {ek}')
        prefixes = ['', 'model.']
        weights_to_load = {}
        state_dict_keys = list(state_dict.keys())

        for internal_key, mapping_dict in self.state_dict_mapping.items():
            model_key = next(iter(mapping_dict))
            external_key = mapping_dict[model_key]
            for pre in prefixes:
                k = pre + external_key
                if k in state_dict_keys:
                    if k in state_dict:
                        weights_to_load[model_key] = state_dict.pop(k).to(torch.float32)
                    else:
                        logger.info(f'  KEY IN SNAPSHOT BUT NOT IN DICT: {k}')
                    break

        result = self.load_state_dict(weights_to_load, strict=False)
        logger.info(f'  loaded {len(weights_to_load)} weights into projector')
        if result.missing_keys:
            logger.warning(f'Missing projector keys: {result.missing_keys}')
        if result.unexpected_keys:
            logger.warning(f'Unexpected projector keys: {result.unexpected_keys}')
        # Log remaining deepstack keys in state_dict
        remaining = [k for k in state_dict if 'deepstack' in k]
        logger.info(f'  remaining deepstack keys in sd: {remaining}')

        # Move to device (stay on CPU if jit_trace mode)
        import os
        num_gpu = torch.cuda.device_count()
        if num_gpu > 0 and not getattr(self, 'jit_trace', False):
            gpu_id = os.getenv('LOCAL_RANK', 0)
            self.device_list = [f'cuda:{gpu_id}']
            self.to(f'cuda:{gpu_id}')
        else:
            self.device_list = ['cpu']

        return self, state_dict

    def forward(self, x):
        """Main merger forward: norm -> reshape -> fc1 -> gelu -> fc2."""
        x = self.norm(x)
        x = x.view(-1, self.hidden_size)
        x = self.linear_fc2(self.act(self.linear_fc1(x)))
        return x

    def _calculate_ptq_fixed_shape_batch(self):
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

    def get_jit_trace_inputs(self):
        self._calculate_ptq_fixed_shape_batch()
        fixed_batch_size = int((self.image_grid_thw[0][1] * self.image_grid_thw[0][2]).item())
        return torch.randn(fixed_batch_size, self.config.embed_dim, device='cpu', dtype=torch.float32)

    def get_ptq_inputs(self, args=None, **kwargs):
        import numpy as np
        if not hasattr(self, 'image_grid_thw'):
            self._calculate_ptq_fixed_shape_batch()
        fixed_batch_size = int((self.image_grid_thw[0][1] * self.image_grid_thw[0][2]).item())
        embed_dim = self.config.embed_dim
        input_shapes = [[fixed_batch_size, embed_dim]]
        input_value_ranges = [None]

        if args is not None and hasattr(args, 'calibration_dataset') and args.calibration_dataset != 'fake':
            import os as _os
            from ...utils import utils as _utils
            def calib_data_gen():
                # Projector calib data is saved under encoder/chunk_{last_encoder_idx+1}
                # Try 'projector' dir first, fallback to encoder chunk
                proj_dir = _os.path.join(args.calibration_dataset, 'projector')
                if not _os.path.exists(proj_dir):
                    # Find the last encoder chunk index + 1
                    enc_dir = _os.path.join(args.calibration_dataset, 'encoder')
                    chunks = [int(d.split('_')[1]) for d in _os.listdir(enc_dir) if d.startswith('chunk_')]
                    proj_dir = _os.path.join(enc_dir, f'chunk_{max(chunks)}')
                for f in _utils.get_sorted_path_list(proj_dir, '.npz', sep='-'):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]
        else:
            def calib_data_gen():
                for _ in range(10):
                    yield [np.random.rand(fixed_batch_size, embed_dim).astype(np.float32)]

        def eval_data_gen():
            return calib_data_gen()

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen
