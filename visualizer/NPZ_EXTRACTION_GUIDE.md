# ImageNet64 NPZ Extraction Guide

## Overview

The `extract_real_imagenet.py` script now supports **both** ImageNet formats:
1. **JPEG directory structure** (original ImageNet with synset directories)
2. **ImageNet64 NPZ format** (efficient batch files with pre-resized 64×64 images)

This guide covers the NPZ format, which is **significantly faster** for ImageNet64 models.

## NPZ Format Structure

ImageNet64 NPZ files contain:
- **`data`**: Images as uint8 arrays `(N, 12288)` - flattened from `(N, 3, 64, 64)`
- **`labels`**: Class labels `(N,)` as 1-indexed integers (1-1000)
- **`mean`**: Dataset mean `(12288,)` for normalization (optional)

### Example Structure
```
data/Imagenet64_train_npz/
├── train_data_batch_1.npz   # ~128K images
├── train_data_batch_2.npz
├── ...
└── train_data_batch_10.npz

Total: ~1.28M training images
```

## Usage

### Basic Example

Extract activations from 1000 samples:

```bash
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 1000 \
  --batch_size 64 \
  --layers encoder_bottleneck,midblock \
  --conditioning_sigma 80.0 \
  --device cuda
```

### Full Example with All Options

```bash
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch_fid1.51_checkpoint_model_193500.pth \
  --npz_dir data/Imagenet64_train_npz \
  --output_dir data \
  --num_samples 10000 \
  --batch_size 128 \
  --layers encoder_bottleneck,midblock \
  --conditioning_sigma 80.0 \
  --seed 42 \
  --device cuda
```

## Command-Line Arguments

### Required
- `--checkpoint_path PATH`: Model checkpoint
- **Either** `--npz_dir PATH` (for NPZ) **OR** `--imagenet_dir PATH` (for JPEG)

### Optional
- `--output_dir PATH`: Output directory (default: `data`)
- `--num_samples N`: Number of samples to process (default: 1000)
- `--batch_size N`: Processing batch size (default: 64)
- `--layers L1,L2`: Layers to extract (default: `encoder_bottleneck,midblock`)
- `--conditioning_sigma SIGMA`: Forward pass sigma (default: 80.0, matches DMD2 training)
- `--seed N`: Random seed for shuffling (default: 10)
- `--device {cuda,mps,cpu}`: Device (default: auto-detect)
- `--adapter NAME`: Model adapter (default: `dmd2-imagenet-64`, also: `edm-imagenet-64`)
- `--label_dropout FLOAT`: Label dropout (default: 0.0, use 0.1 for CFG models)

### Notes
- `--split` is **ignored** for NPZ format (NPZ files are train-only)
- Cannot use both `--npz_dir` and `--imagenet_dir` simultaneously

## NPZ vs JPEG Comparison

| Feature | NPZ Format | JPEG Format |
|---------|------------|-------------|
| **Speed** | ⚡ Very Fast (10-100x) | Slower (disk I/O) |
| **Resolution** | 64×64 only | Any resolution |
| **Labels** | Included (0-999) | From directory structure |
| **Split** | Train only | Train/val |
| **Storage** | ~13 GB (10 files) | ~100+ GB (raw) |
| **Setup** | Download NPZ batches | Full ImageNet dataset |

## Output Structure

Same as JPEG format:

```
data/
├── images/imagenet_real/
│   ├── sample_000000.png  # Original (from NPZ)
│   ├── sample_000001.png
│   └── ...
├── images/imagenet_real_reconstructed/
│   ├── sample_000000.png  # DMD2 reconstructions
│   ├── sample_000001.png
│   └── ...
├── activations/imagenet_real/
│   ├── batch_000000.npz   # Activations
│   ├── batch_000000.json  # Metadata
│   └── ...
└── metadata/imagenet_real/
    └── dataset_info.json   # Global metadata
```

## Metadata Format

### Batch Metadata (`batch_XXXXXX.json`)

```json
{
  "batch_size": 64,
  "layers": ["encoder_bottleneck", "midblock"],
  "samples": [
    {
      "batch_index": 0,
      "class_id": 571,
      "synset_id": "n03804744",
      "class_name": "nail",
      "original_path": "npz_sample_42157"
    },
    ...
  ]
}
```

**Fields**:
- `class_id`: 0-indexed class (0-999)
- `synset_id`: WordNet synset from class labels mapping
- `class_name`: Human-readable class name
- `original_path`: `npz_sample_{global_index}` (for tracking)

## Performance Tips

### Memory Optimization

NPZ format loads images on-demand per batch:

```bash
# Large GPU (24+ GB)
--batch_size 256

# Medium GPU (12-16 GB)
--batch_size 128

# Small GPU (8 GB)
--batch_size 64

# CPU or MPS
--batch_size 32
```

### Processing Speed

Expected throughput:
- **CUDA (V100)**: ~2000-5000 samples/sec
- **MPS (M2 Ultra)**: ~300-500 samples/sec
- **CPU**: ~20-50 samples/sec

### Large-Scale Extraction

Process all 1.28M images:

```bash
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 1281167 \
  --batch_size 256 \
  --device cuda
```

Time estimate: ~10-20 minutes on V100 GPU

## Obtaining ImageNet64 NPZ Files

### Option 1: Download Pre-processed NPZ

ImageNet64 NPZ files are available from:
- [ImageNet downsampled datasets](http://image-net.org/small/download.php)
- Academic mirrors (check ImageNet licensing)

### Option 2: Convert from JPEG

If you have full ImageNet, convert to NPZ:

```python
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def create_imagenet64_npz(imagenet_dir, output_dir, batch_size=128116):
    """Convert ImageNet JPEGs to NPZ format."""
    # Collect all images
    all_images = []
    all_labels = []

    # ... (implementation left as exercise)

    # Save batches
    for batch_idx in range(num_batches):
        np.savez_compressed(
            f"{output_dir}/train_data_batch_{batch_idx+1}.npz",
            data=batch_images,
            labels=batch_labels,
            mean=dataset_mean
        )
```

## Troubleshooting

### Issue: NPZ files not found

**Error**: `No NPZ files found in {npz_dir}`

**Solution**:
```bash
ls data/Imagenet64_train_npz/*.npz
```
Ensure files follow naming pattern: `train_data_batch_*.npz`

### Issue: Label mismatch

**Warning**: Labels in NPZ are 1-indexed but script converts to 0-indexed

**Verification**:
```python
# NPZ label range: 1-1000
# After conversion: 0-999 (correct for DMD2)
```

### Conditioning Sigma

For **real ImageNet** comparison with generated images:
- `--conditioning_sigma 80.0` (default, recommended)
- Matches DMD2 training and generation conditioning
- Ensures real and generated activations are in the same feature space

## Use Cases

### 1. Fast Activation Extraction

Extract activations from 100K real ImageNet samples in minutes:

```bash
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 100000 \
  --batch_size 256 \
  --device cuda
```

### 2. Real vs Generated Distribution Analysis

```bash
# Generate synthetic samples
python generate_dataset.py \
  --model imagenet \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --num_samples 10000

# Extract real ImageNet activations
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 10000

# Compare distributions
python compare_distributions.py  # (future tool)
```

### 3. Class-Balanced Sampling

Select specific samples by class:

```python
# Custom sampling script
import numpy as np

npz_file = "data/Imagenet64_train_npz/train_data_batch_1.npz"
data = np.load(npz_file)
labels = data['labels'] - 1  # 0-indexed

# Get 100 samples of class 0 (tench)
class_0_indices = np.where(labels == 0)[0][:100]
```

## Next Steps

1. **Visualization**: Use `process_embeddings.py` to compute UMAP
2. **Analysis**: Compare real vs generated in embedding space
3. **Metrics**: Compute FID, IS, Precision/Recall

## References

- Main docs: `REAL_IMAGENET_GUIDE.md` (JPEG format)
- Code: `extract_real_imagenet.py`
- Session summary: `Planning/SESSION_REAL_IMAGENET.md`
