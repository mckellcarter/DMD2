# Plan: EDM Visualization Adapter

## Status: IMPLEMENTED (2026-01-12)

## Summary

EDM adapter implemented at `visualizer/adapters/edm_imagenet.py`. Enables visualization of original EDM model activations alongside DMD2 for teacher vs student comparison.

## Implementation

### Files Created/Modified

| File | Status |
|------|--------|
| `visualizer/adapters/edm_imagenet.py` | Created |
| `visualizer/adapters/__init__.py` | Updated (added export) |
| `visualizer/testing/test_adapters.py` | Created (includes EDM tests) |

### Key Features

- `from_pickle()` - loads official EDM `.pkl` checkpoints (supports URLs)
- `from_checkpoint()` - auto-detects format (.pkl vs .pth)
- `forward()` - single denoising step
- `sample()` - full multi-step EDM sampling with Heun solver
- MPS compatibility (float32 fallback for Apple Silicon)
- Same hookable layer names as DMD2 for activation comparison

### Checkpoint

**Original EDM ImageNet-64 from Karras et al.:**
- URL: `https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl`
- Size: ~1.1 GB
- Format: Pickle with `'ema'` key

**Download:**
```bash
bash scripts/download_imagenet.sh checkpoints/
# Or directly:
wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl \
  -O checkpoints/edm-imagenet-64x64-cond-adm.pkl
```

### Usage

```python
from visualizer.adapters import get_adapter

# Load EDM adapter
AdapterCls = get_adapter('edm-imagenet-64')
edm = AdapterCls.from_pickle(
    'checkpoints/edm-imagenet-64x64-cond-adm.pkl',
    device='cuda'  # or 'mps' for Apple Silicon
)

# Multi-step sampling (256 steps, FID=1.36 settings)
images, labels = edm.sample(num_samples=4, class_label=207)

# Single denoising step
denoised = edm.forward(noisy_x, sigma, class_labels)

# Hook activations
def my_hook(module, inp, output):
    print(f"Activation shape: {output.shape}")

edm.register_activation_hooks(['encoder_bottleneck'], my_hook)
```

**Visualizer:**
```bash
python visualizer/visualization_app.py \
  --embeddings data/embeddings/edm_umap.csv \
  --checkpoint_path checkpoints/edm-imagenet-64x64-cond-adm.pkl \
  --adapter edm-imagenet-64
```

## Architecture Reference

### Model Structure (EDMPrecond + DhariwalUNet)

```
Input: (B, 3, 64, 64)
├── Encoder: 64→32→16→8→4, channels 192→384→576→768
├── Bottleneck: (B, 768, 4, 4) with attention
├── Decoder: mirrors encoder with skip connections
└── Output: (B, 3, 64, 64)
```

### Hookable Layers

```python
hookable_layers = [
    'encoder_block_0', ..., 'encoder_block_11',
    'encoder_bottleneck',
    'midblock',
    'decoder_block_1', ..., 'decoder_block_12'
]
```

## DMD2 vs EDM Comparison

| Aspect | DMD2 Adapter | EDM Adapter |
|--------|--------------|-------------|
| Generation | Single-step (distilled) | Multi-step (iterative) |
| `forward()` | One call = final image | One call = one denoise step |
| `sample()` | N/A (use forward) | Full sampling loop |
| Checkpoint | `.pth` / `.bin` | `.pkl` (dnnlib) |
| Speed | ~50x faster | Original diffusion speed |
| Default steps | 1 | 256 (recommended) |

## References

- EDM Paper: https://arxiv.org/abs/2206.00364
- EDM Repo: https://github.com/NVlabs/edm
- Pretrained: https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/
