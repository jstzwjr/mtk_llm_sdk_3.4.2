# Changelog

## Unreleased

- Add `.gitignore` to exclude `__pycache__/` and `*.py[cod]` from version control; remove existing `__pycache__` directories from the working tree.
- Add Qwen3-VL support:
  - New encoder module `models/encoders/{configuration,modeling}_qwen3vl_vision.py` and projector `models/projectors/projector_qwen3_vl_patchmerger.py`.
  - Register `qwen3_vl_vision` / `qwen3_vl` across `utils/const.py` and class resolvers in `utils/utils.py`.
  - Extend `PatchMergerProjectorConfig` with `qwen3_vl` model type and `deepstack_visual_indexes`.
  - `Qwen2_5VLPreEncoder` / `Qwen2_5VLGetEmbeds`: skip window-attention path when `window_size == 0` (Qwen3-VL).
  - `Qwen2VLPreLLM`: emit `visual_pos_masks` (with overture padding) for deepstack injection.
  - `FloatPipeline`: precompute Qwen3-VL 2D positional embeddings, propagate vision rotary embeddings to encoder chunks, capture deepstack intermediates, run postshuffle mergers, and add deepstack embeds at image-token positions in the first N decoder layers (with mask alignment to padded prefill).
  - `QuantizedPipeline` / `inference_quantized.py`: filter encoder model files by `.tflite` / `.mlir` extension instead of excluding `.json`.
- `Qwen2VLGetEmbeds`: cast image embeds to text-embed dtype before scattering to avoid dtype mismatch.
