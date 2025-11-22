# Device Support - DMD2 Visualizer

## Overview

The visualizer supports three execution modes:
1. **Single device** (auto-detect or manual): CUDA / MPS / CPU
2. **Multi-GPU** (Accelerate): CUDA only
3. **Hybrid**: MPS for single device, CUDA for multi-GPU

---

## Device Auto-Detection

**Priority order**: CUDA → MPS → CPU

```python
# In device_utils.py
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
```

**Usage**:
```bash
# Auto-detect (recommended)
python generate_dataset.py --model imagenet --checkpoint_path path/to/ckpt.pth

# Manual override
python generate_dataset.py --device mps ...
```

---

## Single Device Mode

**File**: `generate_dataset.py`

**Supported devices**:
- ✅ CUDA (NVIDIA GPUs)
- ✅ MPS (Apple Silicon M1/M2/M3)
- ✅ CPU (slow, not recommended)

**Features**:
- Device auto-detection via `device_utils.get_device()`
- Manual selection via `--device cuda|mps|cpu`
- MPS fallback to CPU on error
- Device info display (memory, name)

**Example**:
```bash
# MPS (Apple Silicon)
python generate_dataset.py \
  --model imagenet \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --num_samples 1000 \
  --batch_size 32 \
  --device mps

# CUDA (NVIDIA)
python generate_dataset.py \
  --model imagenet \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --num_samples 5000 \
  --batch_size 64 \
  --device cuda
```

---

## Multi-GPU Mode (Accelerate)

**File**: `generate_dataset_accelerate.py`

**Supported devices**:
- ✅ CUDA (multi-GPU distributed)
- ❌ MPS (not supported - Apple Silicon is single-device only)
- ❌ CPU (supported but not recommended)

**Features**:
- Automatic device placement via Accelerator
- Distributed data generation across GPUs
- Process-specific seeds for diversity
- Mixed precision support (fp16, bf16)

**Setup**:
```bash
# One-time configuration
accelerate config
# Answer prompts:
# - Compute environment: This machine
# - Machine type: No distributed training / Multi-GPU
# - Number of GPUs: (auto-detect)
# - Mixed precision: fp16 or no
```

**Example**:
```bash
# Multi-GPU distributed
accelerate launch generate_dataset_accelerate.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --num_samples 10000 \
  --batch_size 64 \
  --mixed_precision fp16

# Single GPU via accelerate (still works)
accelerate launch --num_processes=1 generate_dataset_accelerate.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --num_samples 1000
```

---

## Device-Specific Recommendations

### CUDA (NVIDIA GPUs)

**Best for**: Multi-GPU, large datasets, fastest performance

**Recommended batch sizes**:
- 8 GB VRAM: batch_size=32-64
- 16 GB VRAM: batch_size=64-128
- 24 GB VRAM: batch_size=128+

**Commands**:
```bash
# Single GPU
python generate_dataset.py --device cuda --batch_size 64

# Multi-GPU (2x)
accelerate launch generate_dataset_accelerate.py --batch_size 64
```

---

### MPS (Apple Silicon)

**Best for**: M1/M2/M3 Macs, development, small-medium datasets

**Limitations**:
- Single device only (no multi-GPU)
- Some ops slower than CUDA
- Unified memory (shared with system)

**Recommended batch sizes**:
- M1 (8 GB): batch_size=16-24
- M2 (16 GB): batch_size=24-32
- M3 (24+ GB): batch_size=32-48

**Commands**:
```bash
# MPS (auto-detect)
python generate_dataset.py --model imagenet --batch_size 32

# Explicit MPS
python generate_dataset.py --device mps --batch_size 24
```

**Fallback behavior**:
If MPS placement fails, code automatically falls back to CPU with warning.

---

### CPU

**Best for**: Testing only, no GPU available

**Performance**: ~10-50x slower than GPU

**Not recommended for**:
- Production datasets
- Large samples (>1000)
- Time-sensitive work

**Commands**:
```bash
# Explicit CPU
python generate_dataset.py --device cpu --batch_size 8 --num_samples 100
```

---

## File Comparison

| Feature | generate_dataset.py | generate_dataset_accelerate.py |
|---------|---------------------|--------------------------------|
| **Auto-detect device** | ✅ Yes | ❌ No (Accelerator handles) |
| **Manual device selection** | ✅ Yes (`--device`) | ❌ No |
| **CUDA support** | ✅ Single GPU | ✅ Multi-GPU |
| **MPS support** | ✅ Yes | ❌ No (single-device only) |
| **CPU support** | ✅ Yes | ✅ Yes (not recommended) |
| **Multi-GPU** | ❌ No | ✅ Yes |
| **Mixed precision** | ❌ No | ✅ Yes (fp16, bf16) |
| **Distributed** | ❌ No | ✅ Yes |
| **Best for** | MPS, single GPU | Multi-GPU CUDA |

---

## Decision Tree

```
Do you have multiple NVIDIA GPUs?
├─ YES → Use generate_dataset_accelerate.py
│         accelerate launch ...
│
└─ NO → Do you have Apple Silicon (M1/M2/M3)?
    ├─ YES → Use generate_dataset.py --device mps
    │
    └─ NO → Do you have NVIDIA GPU?
        ├─ YES → Use generate_dataset.py --device cuda
        │
        └─ NO → Use generate_dataset.py --device cpu
                (or get a GPU!)
```

---

## Troubleshooting

### MPS not detected

**Check availability**:
```python
import torch
print(torch.backends.mps.is_available())  # Should be True
print(torch.backends.mps.is_built())      # Should be True
```

**Fix**:
- Update PyTorch: `pip install --upgrade torch torchvision`
- Requires PyTorch 1.12+ on macOS 12.3+

### MPS errors during generation

**Common issues**:
- Out of memory → Reduce `--batch_size`
- Op not implemented → Falls back to CPU automatically
- Slow performance → Some ops are unoptimized on MPS

**Workaround**:
```bash
# Force CPU if MPS has issues
python generate_dataset.py --device cpu --batch_size 8
```

### Multi-GPU not using all GPUs

**Check config**:
```bash
accelerate config  # Reconfigure
accelerate env      # Verify settings
```

**Verify in code**:
```python
# In generate_dataset_accelerate.py output
print(f"Num processes: {accelerator.num_processes}")  # Should match GPU count
```

### CUDA OOM even with small batch

**Solutions**:
1. Reduce batch_size: `--batch_size 16`
2. Use fewer layers: `--layers midblock`
3. Clear GPU cache: `torch.cuda.empty_cache()`
4. Use mixed precision: `--mixed_precision fp16`

---

## Performance Comparison

**ImageNet 64x64, 1000 samples, 2 layers**:

| Device | Batch Size | Time | Notes |
|--------|------------|------|-------|
| RTX 3090 (24GB) | 128 | ~30s | Single GPU |
| RTX 3090 x2 | 64/GPU | ~20s | Distributed |
| M2 Max (32GB) | 32 | ~2m | MPS |
| M1 (8GB) | 16 | ~4m | MPS |
| CPU (16 cores) | 8 | ~45m | Not recommended |

---

## Code References

**Device utilities**: `device_utils.py`
- `get_device()` - Auto-detection
- `get_device_info()` - Device info
- `move_to_device()` - Safe model placement

**Single device**: `generate_dataset.py:80-90`
```python
if device is None:
    device = get_device()

device_info = get_device_info(device)
print(f"Device: {device_info['device']} ({device_info['device_name']})")

generator = move_to_device(generator, device)
```

**Multi-GPU**: `generate_dataset_accelerate.py:60-75`
```python
accelerator = Accelerator(mixed_precision=mixed_precision)
generator = accelerator.prepare(generator)
# Automatic device placement across GPUs
```

---

## Future Enhancements

- [ ] Multi-node distributed (multiple machines)
- [ ] TPU support (Google Cloud)
- [ ] AMD ROCm support (AMD GPUs)
- [ ] Optimized MPS kernels
- [ ] Dynamic batch size adjustment
