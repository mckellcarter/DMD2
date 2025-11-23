# Real ImageNet Activation Extraction - Quick Guide

## Overview

Extract activations from real ImageNet images through the DMD2 model to compare with generated samples. This enables:
- Distribution analysis (real vs generated)
- Mode coverage evaluation
- UMAP visualization in real ImageNet space
- Quality metrics computation

## Prerequisites

1. **DMD2 checkpoint**: Download ImageNet model
2. **ImageNet dataset**: Access to ImageNet validation or training set
3. **Dependencies**: All visualizer requirements installed

## Quick Start

### 1. Prepare ImageNet Data

Ensure your ImageNet directory follows this structure:

```
/path/to/imagenet/
├── val/
│   ├── n01440764/          # Synset directory (tench)
│   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   ├── ILSVRC2012_val_00002138.JPEG
│   │   └── ...
│   ├── n01443537/          # Synset directory (goldfish)
│   │   └── *.JPEG
│   └── ...                 # 1000 synset directories
└── train/
    └── ...
```

### 2. Extract Activations

```bash
cd visualizer

# Extract from 1000 validation images (sigma=0.0 for clean reconstruction)
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch_fid1.51_checkpoint_model_193500.pth \
  --imagenet_dir /path/to/imagenet \
  --num_samples 1000 \
  --batch_size 64 \
  --layers encoder_bottleneck,midblock \
  --split val \
  --device cuda \
  --conditioning_sigma 0.0  # Default: clean reconstruction
```

### 3. Output Files

After processing, you'll have:

```
data/
├── images/
│   ├── imagenet_real/
│   │   ├── sample_000000.JPEG  # Original images
│   │   ├── sample_000001.JPEG
│   │   └── ...
│   └── imagenet_real_reconstructed/
│       ├── sample_000000.png  # DMD2 reconstructions (sigma=0.0)
│       ├── sample_000001.png
│       └── ...
├── activations/imagenet_real/
│   ├── batch_000000.npz    # Batch 0 activations
│   ├── batch_000000.json   # Batch 0 metadata
│   ├── batch_000001.npz
│   ├── batch_000001.json
│   └── ...
└── metadata/imagenet_real/
    └── dataset_info.json   # Global metadata
```

## Command-Line Options

```bash
python extract_real_imagenet.py \
  --checkpoint_path PATH      # Required: DMD2 model checkpoint
  --imagenet_dir PATH         # Required: ImageNet root directory
  --output_dir PATH           # Default: "data"
  --num_samples N             # Default: 1000
  --batch_size N              # Default: 64
  --layers L1,L2,...          # Default: "encoder_bottleneck,midblock"
  --conditioning_sigma SIGMA  # Default: 80.0
  --split {val,train}         # Default: "val"
  --seed N                    # Default: 10
  --device {cuda,mps,cpu}     # Default: auto-detect
```

### Key Parameters

**--num_samples**: Total images to process
- Start with 1000 for quick testing
- Use 10000+ for comprehensive analysis
- Full validation set: 50,000 images

**--batch_size**: Processing batch size
- CUDA (large GPU): 64-128
- MPS (Apple Silicon): 32-64
- CPU: 8-16
- Adjust based on memory

**--layers**: Which layers to extract
- `encoder_bottleneck`: Final encoder output
- `midblock`: Decoder midblock
- `encoder_block_N`: Specific encoder block
- `decoder_block_N`: Specific decoder block

**--split**: ImageNet split
- `val`: Validation set (50k images)
- `train`: Training set (1.2M images)

**--conditioning_sigma**: Forward pass sigma
- **Default: 0.0** (clean reconstruction - captures original ImageNet manifold)
- `0.0`: Model acts as near-identity, minimal transformation
- `80.0`: High noise level (only use if comparing with generated images at same sigma)
- **Why 0.0?** DMD2 is trained with reconstruction loss on real images. At sigma=0.0, the EDM preconditioning makes c_skip=1.0 and c_out=0.0, so output ≈ input. This captures the "original voxel-space manifold" that DMD2 learned to match.

## Metadata Format

### Global Metadata (`dataset_info.json`)

```json
{
  "model_type": "imagenet_real",
  "num_samples": 1000,
  "layers": ["encoder_bottleneck", "midblock"],
  "conditioning_sigma": 80.0,
  "seed": 10,
  "split": "val",
  "class_labels": {...},
  "samples": [...]
}
```

### Batch Metadata (`batch_XXXXXX.json`)

```json
{
  "batch_size": 64,
  "layers": ["encoder_bottleneck", "midblock"],
  "samples": [
    {
      "batch_index": 0,
      "class_id": 0,
      "synset_id": "n01440764",
      "class_name": "tench",
      "original_path": "/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG"
    },
    ...
  ]
}
```

## Loading Activations

### Python Example

```python
import numpy as np
import json
from pathlib import Path

# Load a single batch
batch_path = Path("data/activations/imagenet_real/batch_000000")

# Load activations
activations = np.load(batch_path.with_suffix('.npz'))
encoder_acts = activations['encoder_bottleneck']  # (64, C*H*W)
midblock_acts = activations['midblock']            # (64, C*H*W)

# Load metadata
with open(batch_path.with_suffix('.json')) as f:
    metadata = json.load(f)

# Access sample info
for sample in metadata['samples']:
    print(f"Sample {sample['batch_index']}: {sample['class_name']} ({sample['synset_id']})")

# Extract single sample
sample_idx = 5
sample_encoder = encoder_acts[sample_idx:sample_idx+1]  # (1, C*H*W)
sample_info = metadata['samples'][sample_idx]
print(f"Class: {sample_info['class_name']}, ID: {sample_info['class_id']}")
```

## Common Workflows

### 1. Compare Real vs Generated Distributions

```bash
# Generate synthetic dataset
python generate_dataset.py \
  --model imagenet \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --num_samples 1000 \
  --layers encoder_bottleneck,midblock

# Extract real ImageNet activations
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --imagenet_dir /path/to/imagenet \
  --num_samples 1000 \
  --layers encoder_bottleneck,midblock

# Now you have:
# - data/activations/imagenet/        (generated)
# - data/activations/imagenet_real/   (real)
```

### 2. UMAP on Real Space

```bash
# Extract real activations (larger sample for stable UMAP)
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --imagenet_dir /path/to/imagenet \
  --num_samples 10000 \
  --layers encoder_bottleneck,midblock

# Fit UMAP on real activations
python process_embeddings.py \
  --model imagenet_real \
  --n_neighbors 15 \
  --min_dist 0.1

# Project generated samples (future feature)
# python project_to_real_space.py ...
```

### 3. Class-Balanced Sampling

To get balanced real data across classes, sample manually:

```python
from pathlib import Path
import random

imagenet_val = Path("/path/to/imagenet/val")
synsets = sorted([d for d in imagenet_val.iterdir() if d.is_dir()])

# Sample N images per class
n_per_class = 10
selected = []
for synset_dir in synsets:
    images = list(synset_dir.glob("*.JPEG"))
    selected.extend(random.sample(images, min(n_per_class, len(images))))

# Write to file list
with open("selected_images.txt", "w") as f:
    for img in selected:
        f.write(f"{img}\n")
```

Then modify `extract_real_imagenet.py` to read from this file.

## Troubleshooting

### Issue: Synset directory not found

**Error**: `ImageNet split directory not found: /path/to/imagenet/val`

**Solution**: Ensure ImageNet directory structure is correct:
```bash
ls /path/to/imagenet/val/  # Should show n01440764, n01443537, etc.
```

### Issue: Unknown synset warnings

**Warning**: `Warning: Unknown synset nXXXXXXXX for /path/to/image.JPEG`

**Cause**: Synset ID not in `imagenet_class_labels.json` (may be non-standard synset)

**Impact**: Sample will have `class_id = -1` and `class_name = "unknown"`

### Issue: Out of memory

**Error**: CUDA out of memory

**Solution**: Reduce `--batch_size`:
```bash
--batch_size 32  # or 16, 8
```

### Issue: Slow processing

Processing speed depends on:
- Device (CUDA >> MPS > CPU)
- Batch size (larger = faster per-sample)
- Disk I/O (loading images)

Expected speeds:
- CUDA (V100): ~500 images/sec
- MPS (M2): ~100 images/sec
- CPU: ~10 images/sec

## Next Steps

1. **Visualization**: Integrate with `visualization_app.py` to show real + generated
2. **UMAP Projection**: Fit UMAP on real, project generated samples
3. **Metrics**: Compute FID, Precision, Recall between real and generated
4. **Analysis**: Distribution comparison, mode coverage, class fidelity

## References

- Main documentation: `README.md`
- Session summary: `Planning/SESSION_REAL_IMAGENET.md`
- Future enhancements: `Planning/FUTURE_ENHANCEMENTS.md`
- Code: `extract_real_imagenet.py`
- Tests: `test_extract_real_imagenet.py`
