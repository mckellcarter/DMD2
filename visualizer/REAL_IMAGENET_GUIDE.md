# Real ImageNet Activation Extraction - Quick Guide

## Overview

Extract activations from real ImageNet images through the DMD2 model to compare with generated samples. This enables:
- Distribution analysis (real vs generated)
- Mode coverage evaluation
- UMAP visualization in real ImageNet space
- Quality metrics computation

## Prerequisites

1. **DMD2 checkpoint**: Download ImageNet model
2. **ImageNet dataset**: Either format:
   - **ImageNet64 NPZ** (recommended, 10-100x faster)
   - **JPEG directory structure** (original ImageNet)
3. **Dependencies**: All visualizer requirements installed

## Input Format Options

The script supports **two input formats**:

| Format | Speed | Size | Use Case |
|--------|-------|------|----------|
| **NPZ (Recommended)** | ⚡ Very Fast | 13 GB | ImageNet64 batch processing |
| **JPEG** | Slower | 100+ GB | Full resolution, custom preprocessing |

**See also**: `NPZ_EXTRACTION_GUIDE.md` for comprehensive NPZ documentation

## Quick Start

### Option A: ImageNet64 NPZ Format (Recommended)

**1. Prepare NPZ Files**

Ensure ImageNet64 NPZ files are available:
```
data/Imagenet64_train_npz/
├── train_data_batch_1.npz
├── train_data_batch_2.npz
├── ...
└── train_data_batch_10.npz
```

**2. Extract Activations**

```bash
cd visualizer

# Extract from 10,000 training images 
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_fid1.51.pth  \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 10000 \
  --batch_size 128 \
  --layers encoder_bottleneck,midblock \
  --conditioning_sigma 80.0
```

**Performance**: ~2000-5000 samples/sec on V100 GPU

### Option B: JPEG Directory Format

**1. Prepare ImageNet Directory**

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

**2. Extract Activations**

```bash
cd visualizer

# Extract from 1000 validation images
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --imagenet_dir /path/to/imagenet \
  --num_samples 1000 \
  --batch_size 64 \
  --layers encoder_bottleneck,midblock \
  --split val \
  --device cuda \
  --conditioning_sigma 80.0  # Matches DMD2 training/generation
```

**Performance**: ~100-500 samples/sec (slower due to disk I/O)

### Output Files

After processing (same for both formats), you'll have:

```
data/
├── images/
│   ├── imagenet_real/
│   │   ├── sample_000000.png   # Original images (PNG for NPZ, JPEG for directory)
│   │   ├── sample_000001.png
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
  --npz_dir PATH              # NPZ format: Directory with *.npz files
  --imagenet_dir PATH         # JPEG format: ImageNet root directory
  --output_dir PATH           # Default: "data"
  --num_samples N             # Default: 1000
  --batch_size N              # Default: 64
  --layers L1,L2,...          # Default: "encoder_bottleneck,midblock"
  --conditioning_sigma SIGMA  # Default: 80.0
  --split {val,train}         # Default: "train" (JPEG only, ignored for NPZ)
  --seed N                    # Default: 10
  --device {cuda,mps,cpu}     # Default: auto-detect
```

**Important**: Use **either** `--npz_dir` **OR** `--imagenet_dir`, not both.

### Key Parameters

**--num_samples**: Total images to process
- Start with 1000 for quick testing
- Use 10000+ for comprehensive analysis
- Full validation set: 50,000 images

**--batch_size**: Processing batch size
- **NPZ format**:
  - CUDA (large GPU): 128-256
  - CUDA (medium GPU): 64-128
  - MPS (Apple Silicon): 32-64
  - CPU: 16-32
- **JPEG format**:
  - CUDA (large GPU): 64-128
  - MPS (Apple Silicon): 32-64
  - CPU: 8-16
- Adjust based on memory

**--layers**: Which layers to extract
- `encoder_bottleneck`: Final encoder output
- `midblock`: Decoder midblock
- `encoder_block_N`: Specific encoder block
- `decoder_block_N`: Specific decoder block

**--split**: ImageNet split (**JPEG format only**)
- `val`: Validation set (50k images)
- `train`: Training set (1.2M images)
- **Note**: Ignored when using `--npz_dir` (NPZ is train-only)

**--conditioning_sigma**: Forward pass sigma
- **Default: 80.0** (matches DMD2 training and generation)
- `80.0`: Standard conditioning used in DMD2 training/generation
- This value matches the `conditioning_sigma` used during DMD2 training (train_edm.py default)
- **Why 80.0?** DMD2 generator is trained to produce images from `noise * 80.0` at timestep 80.0. Using the same sigma for real images ensures activations are extracted in the same feature space as generated images, enabling proper comparison.

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
      "original_path": "/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG"  # JPEG format
      // OR
      "original_path": "npz_sample_42157"  # NPZ format
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
  --num_samples 10000 \
  --layers encoder_bottleneck,midblock

# Extract real ImageNet activations (NPZ - fast!)
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 10000 \
  --layers encoder_bottleneck,midblock

# Now you have:
# - data/activations/imagenet/        (generated)
# - data/activations/imagenet_real/   (real)
```

### 2. UMAP on Real Space

```bash
# Extract real activations (NPZ - can do large samples quickly)
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 50000 \
  --batch_size 256 \
  --layers encoder_bottleneck,midblock

# Fit UMAP on real activations
python process_embeddings.py \
  --model imagenet_real \
  --n_neighbors 15 \
  --min_dist 0.1

# Project generated samples (future feature)
# python project_to_real_space.py ...
```

### 3. Class-Balanced Sampling (NPZ Format)

**Built-in class-balanced sampling** enables controlled extraction from specific ImageNet classes:

```bash
# Extract 5000 samples from 100 random classes (~50 samples per class)
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 5000 \
  --num_classes 100 \
  --batch_size 128 \
  --device mps

# Extract from specific classes (e.g., animals: classes 0-9)
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 1000 \
  --target_classes "0,1,2,3,4,5,6,7,8,9" \
  --batch_size 128

# Balanced extraction from all 1000 classes
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 10000 \
  --num_classes 1000 \
  --batch_size 256
```

**Parameters:**
- `--num_classes`: Number of classes to sample from (default: 1000)
- `--target_classes`: Comma-separated class IDs. If not specified, randomly selects `num_classes`

**Algorithm:**
- Calculates `samples_per_class = num_samples // num_classes`
- Iterates through NPZ files sequentially
- Collects samples until each target class has ~`samples_per_class`
- Stops early once quota met (efficient)

**Output:** Prints class distribution statistics (min/max/mean samples per class)

## Troubleshooting

### Issue: Format argument error

**Error**: `Either --imagenet_dir or --npz_dir must be provided`

**Solution**: Specify exactly one input format:
```bash
# NPZ format
--npz_dir data/Imagenet64_train_npz

# OR JPEG format (not both)
--imagenet_dir /path/to/imagenet
```

### Issue: NPZ files not found

**Error**: `No NPZ files found in {npz_dir}`

**Solution**: Check NPZ directory has `*.npz` files:
```bash
ls data/Imagenet64_train_npz/*.npz
# Should show: train_data_batch_1.npz, train_data_batch_2.npz, ...
```

### Issue: Synset directory not found (JPEG format)

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
- **Input format** (NPZ >> JPEG)
- Device (CUDA >> MPS > CPU)
- Batch size (larger = faster per-sample)
- Disk I/O (loading images)

Expected speeds:

**NPZ format**:
- CUDA (V100): ~2000-5000 samples/sec
- MPS (M2 Ultra): ~300-500 samples/sec
- CPU: ~20-50 samples/sec

**JPEG format**:
- CUDA (V100): ~500 samples/sec
- MPS (M2): ~100 samples/sec
- CPU: ~10 samples/sec

**Solution**: Use NPZ format for 10-100x speedup!

## Next Steps

1. **Visualization**: Integrate with `visualization_app.py` to show real + generated
2. **UMAP Projection**: Fit UMAP on real, project generated samples
3. **Metrics**: Compute FID, Precision, Recall between real and generated
4. **Analysis**: Distribution comparison, mode coverage, class fidelity

## Format Comparison

| Feature | NPZ Format | JPEG Format |
|---------|------------|-------------|
| **Speed** | ⚡ 10-100x faster | Baseline |
| **Resolution** | 64×64 only | Any (resized to 64×64) |
| **Labels** | Included in files | From directory structure |
| **Split** | Train only (1.28M) | Train/val (1.2M/50k) |
| **Storage** | 13 GB (10 files) | 100+ GB (raw images) |
| **Setup** | Download NPZ batches | Full ImageNet dataset |
| **Preprocessing** | Already normalized | Load + resize each image |

**Recommendation**: Use **NPZ format** for ImageNet64 models. Only use JPEG if you need:
- Validation set specifically
- Custom preprocessing
- Different resolutions

## References

- **NPZ guide**: `NPZ_EXTRACTION_GUIDE.md` (comprehensive NPZ documentation)
- Main documentation: `README.md`
- Session summary (JPEG): `Planning/SESSION_REAL_IMAGENET.md`
- Session summary (NPZ): `Planning/SESSION_NPZ_SUPPORT.md`
- Future enhancements: `Planning/FUTURE_ENHANCEMENTS.md`
- Code: `extract_real_imagenet.py`
- Tests: `test_extract_real_imagenet.py`
