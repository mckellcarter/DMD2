# Plan: Stable Diffusion Visualization Adapters

## Status: PLANNED

## Summary

Adapters for Stable Diffusion models (v1.5 and SDXL) to enable DMD2 visualizer support for text-to-image diffusion models. Unlike class-conditional ImageNet models, SD uses CLIP text embeddings for conditioning.

## Model Variants

| Variant | Resolution | Text Encoder | Model ID |
|---------|------------|--------------|----------|
| SD v1.5 | 512x512 | CLIP ViT-L/14 (single) | `runwayml/stable-diffusion-v1-5` |
| SDXL | 1024x1024 | CLIP ViT-L/14 + OpenCLIP ViT-bigG (dual) | `stabilityai/stable-diffusion-xl-base-1.0` |

## Architecture Reference

### UNet2DConditionModel Layer Structure

```
Input: (B, 4, H/8, W/8) latent space
├── conv_in
├── down_blocks[0]: CrossAttnDownBlock2D (320 ch)
├── down_blocks[1]: CrossAttnDownBlock2D (640 ch)
├── down_blocks[2]: CrossAttnDownBlock2D (1280 ch)
├── down_blocks[3]: DownBlock2D (1280 ch, no cross-attn)
├── mid_block: UNetMidBlock2DCrossAttn (1280 ch)
├── up_blocks[0]: UpBlock2D (1280 ch)
├── up_blocks[1]: CrossAttnUpBlock2D (1280 ch)
├── up_blocks[2]: CrossAttnUpBlock2D (640 ch)
├── up_blocks[3]: CrossAttnUpBlock2D (320 ch)
├── conv_norm_out + conv_out
└── Output: (B, 4, H/8, W/8)
```

**Note:** SDXL has additional blocks and higher channel counts.

### Hookable Layer Naming Convention

```python
hookable_layers = [
    'down_block_0', 'down_block_1', 'down_block_2', 'down_block_3',
    'mid_block',
    'up_block_0', 'up_block_1', 'up_block_2', 'up_block_3'
]
```

Layer access pattern:
```python
unet = model.feedforward_model if hasattr(model, 'feedforward_model') else model
unet.down_blocks[idx]  # down_block_N
unet.mid_block         # mid_block
unet.up_blocks[idx]    # up_block_N
```

## Key Differences from ImageNet Adapters

| Aspect | ImageNet (EDM/DMD2) | Stable Diffusion |
|--------|---------------------|------------------|
| Conditioning | Class labels (one-hot, 1000 classes) | Text embeddings (CLIP) |
| Resolution | 64x64 pixels | 512x512 (v1.5) or 1024x1024 (SDXL) |
| Working space | Pixel space | Latent space (VAE encoded) |
| Checkpoint format | `.pkl` or `.pth` | diffusers format, `.safetensors` |
| Architecture | DhariwalUNet (EDM) | UNet2DConditionModel (diffusers) |
| Wrapper | EDMPrecond | May be wrapped in `feedforward_model` |
| num_classes | 1000 | 0 (text-conditioned) |
| Extra conditioning | None | SDXL: `time_ids`, `text_embeds` |

## Implementation Considerations

### Adapter Design Decision

**Option A: Unified SD Adapter (Recommended)**
- Single `SDAdapter` class handling both v1.5 and SDXL
- Variant detected from model config or explicit parameter
- Pro: Less code duplication, easier maintenance
- Con: Slightly more complex internal logic

**Option B: Separate Adapters**
- `SDv15Adapter` and `SDXLAdapter` as separate classes
- Pro: Cleaner separation, variant-specific optimizations
- Con: Code duplication for shared logic

**Recommendation:** Option A with variant parameter.

### Forward Pass Interface

```python
def forward(
    self,
    x: torch.Tensor,              # Noisy latent (B, 4, H/8, W/8)
    timestep: torch.Tensor,       # Discrete timestep (0-999)
    prompt_embeds: torch.Tensor,  # Text embeddings from CLIP
    added_cond_kwargs: dict = None,  # SDXL only: time_ids, text_embeds
    **kwargs
) -> torch.Tensor:
```

## Activation Shape Reference

### SDXL (1024x1024)

| Layer | Shape (C, H, W) | Notes |
|-------|-----------------|-------|
| down_block_0 | (320, 128, 128) | After first downblock |
| down_block_1 | (640, 64, 64) | |
| down_block_2 | (1280, 32, 32) | |
| down_block_3 | (1280, 16, 16) | No cross-attention |
| mid_block | (1280, 16, 16) | Bottleneck |
| up_block_0 | (1280, 16, 16) | |
| up_block_1 | (1280, 32, 32) | |
| up_block_2 | (640, 64, 64) | |
| up_block_3 | (320, 128, 128) | |

### SD v1.5 (512x512)

| Layer | Shape (C, H, W) | Notes |
|-------|-----------------|-------|
| down_block_0 | (320, 64, 64) | |
| down_block_1 | (640, 32, 32) | |
| down_block_2 | (1280, 16, 16) | |
| down_block_3 | (1280, 8, 8) | |
| mid_block | (1280, 8, 8) | Bottleneck |
| up_block_0 | (1280, 8, 8) | |
| up_block_1 | (1280, 16, 16) | |
| up_block_2 | (640, 32, 32) | |
| up_block_3 | (320, 64, 64) | |

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `visualizer/adapters/sd.py` | Create | Main SD adapter implementation |
| `visualizer/adapters/__init__.py` | Update | Export new adapter |
| `visualizer/testing/test_sd_adapter.py` | Create | Unit tests |

## Usage Examples

```python
from visualizer.adapters import get_adapter

# Load SDXL adapter
AdapterCls = get_adapter('sdxl')
sdxl = AdapterCls.from_pretrained(device='cuda')

# Or load DMD2-trained checkpoint
sdxl = AdapterCls.from_checkpoint(
    'checkpoints/sdxl_dmd2.bin',
    device='cuda'
)

# Hook activations
def my_hook(module, inp, output):
    print(f"mid_block shape: {output.shape}")

sdxl.register_activation_hooks(['mid_block'], my_hook)
```

## Integration with Existing DMD2 Code

Reference files for SD training patterns:
- `main/sd_unified_model.py` - Model wrapper with `feedforward_model`
- `main/sd_unet_forward.py` - Modified forward for classification
- `main/sdxl/sdxl_text_encoder.py` - Dual text encoder
- `demo/text_to_image_sdxl.py` - Inference example

## References

- Diffusers UNet: https://huggingface.co/docs/diffusers/api/models/unet2d-cond
- SDXL Paper: https://arxiv.org/abs/2307.01952
- SD v1.5: https://huggingface.co/runwayml/stable-diffusion-v1-5
- DMD2 SD Training: `main/train_sd.py`
