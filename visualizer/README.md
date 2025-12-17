# DMD2 Visualizer

Interactive visualization tool for DMD2 model activations using UMAP dimensionality reduction.

## Overview

Visualize UNet activations from DMD2 models (ImageNet, SDXL, SDv1.5) in an interactive 2D space. Based on [MNIST_Diff_Viz](https://github.com/mckellcarter/MNIST_Diff_Viz).

## Features

- **Multi-model support**: ImageNet-64x64, SDXL-1024, SDv1.5-512
- **Layer selection**: Visualize different UNet layers (encoder, midblock, decoder)
- **Interactive UMAP**: Adjust parameters in real-time
- **Hover previews**: See generated images on mouseover
- **Color coding**: By class label (ImageNet) or semantic clusters
- **Neighbor selection**: Click to toggle manual neighbors, view K-nearest neighbors
- **Class labels**: Human-readable ImageNet class names
- **Resume generation**: Incrementally append to existing datasets
- **Generate from neighbors**: Create new images by interpolating neighbor activations via UMAP inverse transform with masked generation

## Installation

```bash
# Install main DMD2 dependencies first
cd ..
pip install -r requirements.txt

# Install visualizer dependencies
cd visualizer
pip install -r requirements.txt
```

## Quick Start

### 1. Download Pretrained Model

```bash
# ImageNet FID 1.51 checkpoint
bash ../scripts/download_hf_checkpoint.sh \
  "imagenet/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch_fid1.51_checkpoint_model_193500" \
  ../checkpoints/
```

### 2. Generate Dataset

```bash
# Single GPU/MPS (auto-detects device)
python generate_dataset.py \
  --model imagenet \
  --checkpoint_path ../checkpoints/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch_fid1.51_checkpoint_model_193500.pth \
  --num_samples 1000 \
  --batch_size 64 \
  --layers encoder_bottleneck,midblock

# Explicitly specify MPS (Apple Silicon)
python generate_dataset.py \
  --model imagenet \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --num_samples 1000 \
  --batch_size 32 \
  --device mps

# Resume/append to existing dataset
# Only generates the difference (5000 - existing_count)
python generate_dataset.py \
  --model imagenet \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --num_samples 5000 \
  --batch_size 32 \
  --device mps

# Multi-GPU with Accelerate
accelerate launch generate_dataset_accelerate.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --num_samples 5000 \
  --batch_size 64 \
  --layers encoder_bottleneck,midblock
```

This creates:
- `data/images/imagenet/` - Generated images
- `data/activations/imagenet/` - UNet activations (NPZ)
- `data/metadata/imagenet/` - Dataset info (JSON)

**Resume capability**: Re-run with higher `--num_samples` to append. Existing samples are detected and skipped.

### 3. Process Embeddings

```bash
# Compute UMAP projection
python process_embeddings.py \
  --model imagenet \
  --n_neighbors 15 \
  --min_dist 0.1
```

Creates `data/embeddings/imagenet_umap_n15_d0.1.csv`

### 4. Launch Visualizer

```bash
# With precomputed embeddings
python visualization_app.py \
  --embeddings data/embeddings/imagenet_umap_n15_d0.1.csv

# With generation enabled (requires checkpoint)
python visualization_app.py \
  --embeddings data/embeddings/imagenet_umap_n15_d0.1.csv \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --device cuda

# Or load activations for dynamic UMAP
python visualization_app.py --data_dir data
```

Open http://localhost:8050

## Usage

### Device Selection

**Auto-detection** (default):
```bash
python generate_dataset.py --model imagenet --checkpoint_path path/to/ckpt.pth
# Automatically uses: CUDA > MPS > CPU
```

**Manual selection**:
```bash
# Apple Silicon MPS
python generate_dataset.py --device mps ...

# CUDA
python generate_dataset.py --device cuda ...

# CPU (slow)
python generate_dataset.py --device cpu ...
```

**Multi-GPU** (Accelerate):
```bash
# Configure once
accelerate config

# Run distributed
accelerate launch generate_dataset_accelerate.py \
  --checkpoint_path path/to/ckpt.pth \
  --num_samples 10000 \
  --batch_size 64
```

### Generate Dataset Options

```bash
python generate_dataset.py \
  --model imagenet \
  --checkpoint_path /path/to/checkpoint.pth \
  --num_samples 5000 \
  --batch_size 64 \
  --layers encoder_bottleneck,midblock \
  --samples_per_class 5 \  # Optional: balanced dataset
  --conditioning_sigma 80.0 \
  --seed 10 \
  --device mps  # Optional: auto-detect if omitted
```

**Layers**:
- ImageNet: `encoder_bottleneck`, `midblock`, `encoder_block_N`, `decoder_block_N`
- SDXL/SDv1.5: `down_block_N`, `mid_block`, `up_block_N`

### Process Embeddings Options

```bash
python process_embeddings.py \
  --model imagenet \
  --n_neighbors 50 \  # 5-100, controls local vs global structure
  --min_dist 0.05 \   # 0.0-1.0, controls point spread
  --metric euclidean \
  --max_samples 10000
```

### Visualizer Controls

**UI Controls**:
- Model selector: Switch between ImageNet/SDXL/SDv1.5
- n_neighbors slider: Adjust UMAP neighborhood size
- min_dist slider: Adjust UMAP point spread
- Recalculate button: Recompute UMAP with new parameters
- Export button: Download embeddings CSV

**Interaction**:
- Hover over points to see images and class labels
- Click to select point (blue highlight)
- Click "Find Neighbors" to show K-nearest neighbors (red thin ring)
- Click other points to toggle manual neighbors (red thick ring)
- Click "Generate Image" to create new image from neighbor center activation
- Click "✕" to clear selection and neighbors
- Zoom/pan the plot

**Generation from Neighbors** (requires `--checkpoint_path`):
1. Select a point on the UMAP plot
2. Find K-nearest neighbors or manually select neighbors by clicking
3. Click "Generate Image" button
4. The system will:
   - Calculate the center of neighbors in 2D UMAP space
   - Use UMAP inverse_transform to map back to activation space
   - Set a hook mask to hold the layer output constant
   - Generate a new image with the masked activation
   - Display the new image as a green star at the neighbor center
5. Generated images are saved to the dataset and can be used as new selection points

## Data Structure

```
data/
├── images/
│   └── imagenet/
│       └── sample_000000.png
├── activations/
│   └── imagenet/
│       ├── sample_000000.npz  # Compressed activations
│       └── sample_000000.json # Metadata
├── embeddings/
│   ├── imagenet_umap_n15_d0.1.csv  # UMAP coordinates + metadata
│   └── imagenet_umap_n15_d0.1.json # UMAP parameters
└── metadata/
    └── imagenet/
        └── dataset_info.json  # Global dataset info
```

## Pretrained Models

All models available at [tianweiy/DMD2](https://huggingface.co/tianweiy/DMD2):

**ImageNet-64x64**:
- `imagenet_gan_classifier_genloss3e-3_diffusion1000_lr5e-7_resume_fid1.28_checkpoint_model_548000` (FID 1.28)
- `imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch_fid1.51_checkpoint_model_193500` (FID 1.51)

**SDXL-1024** (not yet implemented):
- `sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch_checkpoint_model_019000` (4-step)

**SDv1.5-512** (not yet implemented):
- `laion6.25_sd_baseline_8node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_resume_fid8.35_checkpoint_model_041000`

## Real ImageNet Activation Extraction

Extract activations from real ImageNet images to compare with generated samples.

**Two input formats supported**:
1. **ImageNet64 NPZ** (recommended, 10-100x faster)
2. **JPEG directory structure** (original ImageNet)

### Using ImageNet64 NPZ Format (Recommended)

```bash
# Extract activations from ImageNet64 NPZ files
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 10000 \
  --batch_size 128 \
  --layers encoder_bottleneck,midblock \
  --device cuda \
  --conditioning_sigma 80.0  # Matches DMD2 training/generation
```

**NPZ format details**: See `NPZ_EXTRACTION_GUIDE.md` for full documentation.

### Using JPEG Format

```bash
# Extract activations from ImageNet JPEG directory
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

**Conditioning Sigma**:
- `--conditioning_sigma 80.0` (default): Matches DMD2 training and generation conditioning
- This ensures real and generated image activations are extracted in the same feature space
- Value matches the default used in `train_edm.py` and `imagenet_example.py`

**Expected ImageNet structure**:
```
imagenet/
├── val/
│   ├── n01440764/
│   │   └── ILSVRC2012_val_*.JPEG
│   ├── n01443537/
│   │   └── ILSVRC2012_val_*.JPEG
│   └── ...
└── train/
    └── ...
```

**Output structure**:
```
data/
├── images/
│   ├── imagenet_real/
│   │   └── sample_000000.JPEG (copied from original)
│   └── imagenet_real_reconstructed/
│       └── sample_000000.png (reconstructed by DMD2 at sigma=0.0)
├── activations/imagenet_real/
│   ├── batch_000000.npz  # Batch activations
│   └── batch_000000.json # Batch metadata with ImageNet IDs
└── metadata/imagenet_real/
    └── dataset_info.json  # Global metadata
```

**Reconstructed Images**: The `imagenet_real_reconstructed` directory contains images processed through DMD2 at the specified conditioning sigma. These are useful for qualitative evaluation of the model's reconstruction/generation behavior.

**Batch metadata format**:
Each batch JSON file includes ImageNet identifiers:
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
      "original_path": "/path/to/imagenet/val/n01440764/img.JPEG"
    },
    ...
  ]
}
```

**Use cases**:
- Compare real vs generated activation distributions
- Fit UMAP on real ImageNet, project generated samples
- Analyze mode coverage and distribution shift
- Visualize both real and generated in same embedding space

## Architecture

**extract_activations.py**: Hook-based activation capture
**generate_dataset.py**: Batch image + activation generation (for generated images)
**extract_real_imagenet.py**: Batch activation extraction from real ImageNet images
**process_embeddings.py**: UMAP dimensionality reduction
**visualization_app.py**: Dash/Plotly interactive app

## Troubleshooting

**Out of memory during generation**:
- Reduce `--batch_size`
- MPS (Apple Silicon): Try batch_size=32 or 16
- Generate in chunks with different seeds

**MPS-specific issues**:
- If MPS fails, code falls back to CPU automatically
- Some ops may be slower on MPS vs CUDA
- Reduce batch size for M1/M2 (8-16 GB unified memory)

**Multi-GPU not working**:
- Run `accelerate config` first
- Ensure all GPUs are CUDA (MPS doesn't support multi-device)
- Use `generate_dataset_accelerate.py` (not `generate_dataset.py`)

**UMAP too slow**:
- Reduce `--max_samples` when processing
- Use fewer layers during generation

**Missing images in visualizer**:
- Check `data/images/{model}/` exists
- Verify paths in embeddings CSV are relative to `data_dir`

**Device not detected**:
- Check `torch.backends.mps.is_available()` for MPS
- Check `torch.cuda.is_available()` for CUDA
- Update PyTorch to latest version

## Future Enhancements

See `Planning/FUTURE_ENHANCEMENTS.md` for detailed proposals.

**High priority**:
- Label conditioning options (uniform, zero, neighbor-average)
- Additional UI improvements

## References

- [MNIST_Diff_Viz](https://github.com/mckellcarter/MNIST_Diff_Viz)
- [DMD2 Paper](https://arxiv.org/abs/2405.14867)
- [DMD2 Repo](https://github.com/tianweiy/DMD2)
- [UMAP](https://umap-learn.readthedocs.io/)
