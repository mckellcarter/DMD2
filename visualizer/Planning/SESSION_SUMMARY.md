# Session Summary - DMD2 Visualizer (UPDATED)

## What We Completed

### Phase 1: Research & Planning (Previous Session)
1. ✅ **Explored DMD2 codebase** - Understand structure, models, training pipeline
2. ✅ **Identified all model architectures**:
   - ImageNet: `DhariwalUNet` (third_party/edm/training/networks.py:373)
   - SDXL/SDv1.5: `UNet2DConditionModel` (from diffusers)
3. ✅ **Cataloged pretrained models** - All open on HuggingFace (tianweiy/DMD2)
4. ✅ **Reviewed MNIST_Diff_Viz** - Dash/Plotly UMAP visualizer with neighbor search
5. ✅ **Created comprehensive plan** - See Planning/DMD2_Visualizer_Plan.md

### Phase 2: Full Implementation (Current Session)
6. ✅ **Implemented extract_activations.py** (visualizer/extract_activations.py:1)
   - Hook-based activation capture
   - Support for DhariwalUNet & UNet2DConditionModel
   - NPZ compression, metadata tracking

7. ✅ **Implemented generate_dataset.py** (visualizer/generate_dataset.py:1)
   - ImageNet fully functional
   - Batch generation with progress bars
   - Balanced/random sampling options

8. ✅ **Implemented process_embeddings.py** (visualizer/process_embeddings.py:1)
   - UMAP dimensionality reduction
   - Configurable parameters
   - CSV output with metadata

9. ✅ **Implemented visualization_app.py** (visualizer/visualization_app.py:1)
   - Full Dash/Plotly interface
   - **Neighbor selection features** (like MNIST_Diff_Viz)
   - Real-time UMAP recalculation
   - Image hover/click previews

10. ✅ **Created documentation**:
    - README.md - Complete usage guide
    - QUICKSTART.md - 1-minute setup
    - RESOURCE_REQUIREMENTS.md - Memory/disk estimates
    - run_visualizer.sh - Launch script

## Current Status

**Status**: ✅ **IMPLEMENTATION COMPLETE - READY FOR TESTING**

## Key Questions Answered (Current Session)

### 1. Memory & Disk Requirements

**ImageNet (64x64)**:
- Per sample: ~210 KB (192 KB activations + 10 KB image)
- 1,000 samples: ~203 MB
- 10,000 samples: ~2.0 GB
- GPU (batch 64): ~1.5 GB

**SDXL (1024x1024, mid_block)**:
- Per sample: ~3.5 MB
- 1,000 samples: ~3.6 GB
- GPU (batch 4): ~8-10 GB

**Full breakdown**: See visualizer/RESOURCE_REQUIREMENTS.md

### 2. Data Selection Strategy

**Generation** (generate_dataset.py):
- NOT randomly selected on load
- Generated sequentially during dataset creation
- Options:
  - Random labels: `np.random.randint(0, 1000, num_samples)`
  - Balanced: `--samples_per_class N` (N per class)
- Reproducible via `--seed`

**Visualization**:
- Loads ALL samples from dataset
- Can limit with `--max_samples` in processing step
- Color-coded by class for ImageNet

### 3. Neighbor Selection Features

**Status**: ✅ IMPLEMENTED (like MNIST_Diff_Viz)

**Features**:
- Click point to select
- K-nearest neighbors (sklearn.neighbors.NearestNeighbors)
- Adjustable k (1-20 slider)
- Visual highlighting (red circles)
- Neighbor list with 64×64 thumbnails + distances
- KNN fitted on UMAP coordinates (Euclidean distance)

**Location**: visualizer/visualization_app.py:241-487

## Directory Structure (Final)

```
visualizer/
├── extract_activations.py      # Activation capture (hook-based)
├── generate_dataset.py          # Dataset generation pipeline
├── process_embeddings.py        # UMAP processing
├── visualization_app.py         # Dash app (with neighbor search)
├── requirements.txt             # Viz dependencies
├── README.md                    # Complete guide
├── QUICKSTART.md                # 1-min setup
├── RESOURCE_REQUIREMENTS.md     # Memory/disk estimates
├── run_visualizer.sh            # Launch script
└── data/
    ├── images/imagenet/         # Generated images (PNG)
    ├── activations/imagenet/    # UNet activations (NPZ)
    ├── embeddings/              # UMAP coords (CSV)
    └── metadata/imagenet/       # Dataset info (JSON)
```

## Immediate Next Steps (TESTING PHASE)

1. **Install dependencies**:
   ```bash
   cd visualizer
   pip install -r requirements.txt
   ```

2. **Download ImageNet checkpoint**:
   ```bash
   cd ..
   bash scripts/download_hf_checkpoint.sh \
     "imagenet/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch_fid1.51_checkpoint_model_193500" \
     checkpoints/
   ```

3. **Test small dataset** (100 samples):
   ```bash
   cd visualizer
   python generate_dataset.py \
     --model imagenet \
     --checkpoint_path ../checkpoints/imagenet_*.pth \
     --num_samples 100 \
     --batch_size 50
   ```

4. **Process UMAP**:
   ```bash
   python process_embeddings.py --model imagenet
   ```

5. **Launch visualizer**:
   ```bash
   ./run_visualizer.sh imagenet
   ```

6. **Test neighbor search**:
   - Click any point
   - Adjust k-neighbors slider
   - Click "Find Neighbors"
   - Verify highlighting and thumbnails

## Environment Notes

- User will restart from installed environment
- Main requirements.txt already has: torch, diffusers, transformers, etc.
- Need to add visualizer-specific deps

## Key Code Locations for Reference

**ImageNet**:
- Model wrapper: `main/edm/edm_unified_model.py`
- Network: `third_party/edm/training/networks.py:373` (DhariwalUNet)
- Demo: `demo/imagenet_example.py`

**SDXL**:
- Model wrapper: `main/sd_unified_model.py`
- Network: From diffusers library (UNet2DConditionModel)
- Demo: `demo/text_to_image_sdxl.py`

**Checkpoints**:
- Download script: `scripts/download_hf_checkpoint.sh`
- Example: `bash scripts/download_hf_checkpoint.sh "imagenet/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch_fid1.51_checkpoint_model_193500" checkpoints/`

## Resume Point

When resuming:
1. Verify environment has main DMD2 dependencies installed
2. Start implementing `visualizer/extract_activations.py`
3. Focus on ImageNet first (simpler architecture)
4. Use PyTorch hooks to capture intermediate activations
