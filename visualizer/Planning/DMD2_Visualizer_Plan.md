# DMD2 Visualizer Plan

## Project Overview
Interactive visualization tool for DMD2 (Improved Distribution Matching Distillation) model activations, similar to MNIST_Diff_Viz but extended for text-to-image and class-conditional generation.

---

## 1. Model Architectures & Subprojects

### ImageNet-64x64
- **Architecture**: DhariwalUNet (from NVLabs' EDM)
- **Location**: `third_party/edm/training/networks.py:373`
- **Type**: Class-conditional generator (1000 ImageNet classes)
- **Resolution**: 64x64
- **Wrapper**: `main/edm/edm_unified_model.py`

### SDXL (1024x1024)
- **Architecture**: UNet2DConditionModel (from Diffusers)
- **Location**: Imported from HuggingFace Diffusers library
- **Type**: Text-to-image, dual text encoders (CLIP-L + OpenCLIP-G)
- **Wrapper**: `main/sd_unified_model.py`
- **Variants**: Full UNet + LoRA adapters

### SDv1.5 (512x512)
- **Architecture**: UNet2DConditionModel (from Diffusers)
- **Location**: Same as SDXL
- **Type**: Text-to-image, single CLIP encoder
- **Wrapper**: `main/sd_unified_model.py`

**Common Pattern**: All models use a discriminator/guidance component with dual UNets (real frozen, fake trainable) + GAN loss.

---

## 2. Pretrained Models Availability

### ImageNet (Huggingface: `tianweiy/DMD2`)

| Model | FID | Status | Link |
|-------|-----|--------|------|
| 1-step (best) | 1.28 | ✅ Open | [HF Link](https://huggingface.co/tianweiy/DMD2/tree/main/model/imagenet/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr5e-7_resume_fid1.28_checkpoint_model_548000) |
| 1-step (base) | 1.51 | ✅ Open | [HF Link](https://huggingface.co/tianweiy/DMD2/tree/main/model/imagenet/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch_fid1.51_checkpoint_model_193500) |
| 1-step (no GAN) | 2.61 | ✅ Open | [HF Link](https://huggingface.co/tianweiy/DMD2/tree/main/model/imagenet/imagenet_lr2e-6_scratch_fid2.61_checkpoint_model_405500) |

### SDXL (Huggingface: `tianweiy/DMD2`)

| Model | FID | Status | Link |
|-------|-----|--------|------|
| 4-step UNet | 19.32 | ✅ Open | [HF Link](https://huggingface.co/tianweiy/DMD2/tree/main/model/sdxl/sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch_checkpoint_model_019000) |
| 4-step LoRA | 19.68 | ✅ Open | [HF Link](https://huggingface.co/tianweiy/DMD2/tree/main/model/sdxl/sdxl_cond999_8node_lr5e-5_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch_lora_checkpoint_model_016000) |
| 1-step UNet | 19.01 | ✅ Open | [HF Link](https://huggingface.co/tianweiy/DMD2/tree/main/model/sdxl/sdxl_cond399_8node_lr5e-7_1step_diffusion1000_gan5e-3_guidance8_noinit_noode_checkpoint_model_024000) |

### SDv1.5 (Huggingface: `tianweiy/DMD2`)

| Model | FID | Status | Link |
|-------|-----|--------|------|
| 1-step (best) | 8.35 | ✅ Open | [HF Link](https://huggingface.co/tianweiy/DMD2/tree/main/model/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_resume_fid8.35_checkpoint_model_041000) |
| 1-step (base) | 9.28 | ✅ Open | [HF Link](https://huggingface.co/tianweiy/DMD2/tree/main/model/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch_fid9.28_checkpoint_model_039000) |

**Status**: All pretrained models are open and hosted on HuggingFace - No NVIDIA-locked models found!

---

## 3. DMD2 Visualizer Architecture

Based on MNIST_Diff_Viz implementation pattern.

### Directory Structure
```
DMD2_Visualizer/
├── visualization_app.py       # Main Dash/Plotly app
├── extract_embeddings.py      # Extract UNet activations during inference
├── process_embeddings.py      # UMAP dimensionality reduction
├── generate_dataset.py        # Generate samples for visualization
├── requirements.txt           # Dependencies
├── run_visualizer.sh          # Launch script
└── data/
    ├── embeddings/            # UMAP coordinates + metadata (CSV)
    ├── images/                # Generated images (PNG/JPG)
    ├── activations/           # Raw UNet activations (NPY)
    └── metadata/              # Prompts, labels, timesteps (JSON)
```

### Key Features

1. **Multi-model Support**
   - Toggle between ImageNet/SDXL/SDv1.5
   - Model switcher in UI
   - Separate data directories per model

2. **Layer Selection**
   - Visualize different UNet layers:
     - Down blocks (early features)
     - Mid block (bottleneck representations)
     - Up blocks (reconstructed features)
   - Dropdown selector in UI

3. **Interactive UMAP**
   - Adjustable parameters:
     - n_neighbors (5-100, default: 15)
     - min_dist (0.0-1.0, default: 0.1)
   - Real-time recalculation
   - Progress indicators

4. **Hover Previews**
   - Show generated images on mouseover
   - Display metadata:
     - ImageNet: Class label
     - SDXL/SDv1.5: Text prompt
     - Timestep (if multi-step)

5. **Color Coding Strategies**
   - **ImageNet**: By class label (1000 classes)
   - **SDXL/SDv1.5**:
     - Option 1: By prompt embeddings cluster
     - Option 2: By semantic category (manual annotation)
     - Option 3: By conditioning timestep

6. **Timestep Visualization** (Multi-step models)
   - Show how embeddings evolve across denoising steps
   - Animation slider
   - Side-by-side comparison

---

## 4. Implementation Steps

### Phase 1: Activation Extraction

**File**: `extract_embeddings.py`

**Functionality**:
- Hook into UNet forward passes
- Save activations from key layers
- Support all three model types

**Hook Locations**:
- **ImageNet**: `main/edm/edm_unified_model.py:39` (feedforward_model)
- **SDXL/SDv1.5**: `main/sd_unified_model.py:39` (feedforward_model)

**Implementation**:
```python
# Register forward hooks on target layers
# Save layer outputs during inference
# Support batch processing
# Handle different layer naming conventions (EDM vs Diffusers)
```

**Output**:
- `data/activations/{model}_{layer}_{sample_id}.npy`
- Metadata JSON with sample info

---

### Phase 2: Dataset Generation

**File**: `generate_dataset.py`

**Functionality**:
- Generate diverse samples for each model
- Save images + activations
- Create metadata

**Dataset Specs**:
- **ImageNet**:
  - 10-50 samples per class (configurable)
  - Total: 10,000-50,000 samples
  - Random seeds for diversity

- **SDXL**:
  - Use COCO prompts (2,000-10,000)
  - Or custom prompt dataset
  - Multiple seeds per prompt

- **SDv1.5**:
  - Similar to SDXL
  - Lower resolution (512x512)

**CLI Interface**:
```bash
python generate_dataset.py \
  --model imagenet \
  --checkpoint_path /path/to/ckpt \
  --num_samples 10000 \
  --output_dir data/ \
  --layers "midblock,upblock1"
```

---

### Phase 3: UMAP Processing

**File**: `process_embeddings.py`

**Functionality**:
- Load raw activations
- Flatten to 1D vectors
- Apply UMAP projection
- Save coordinates + metadata

**UMAP Parameters**:
- Default: `n_neighbors=15, min_dist=0.1, metric='euclidean'`
- Support custom parameters via CLI
- Cache results for different parameter sets

**Output**:
- `data/embeddings/{model}_{layer}_umap.csv`
  - Columns: sample_id, x, y, label/prompt, image_path

**CLI Interface**:
```bash
python process_embeddings.py \
  --model imagenet \
  --layer midblock \
  --n_neighbors 15 \
  --min_dist 0.1 \
  --output_dir data/embeddings/
```

---

### Phase 4: Dash Application

**File**: `visualization_app.py`

**UI Components**:

1. **Control Panel** (left sidebar):
   - Model selector (dropdown)
   - Layer selector (dropdown)
   - UMAP parameters (sliders)
   - Recalculate button
   - Export button

2. **Main Visualization** (center):
   - Plotly scatter plot
   - Interactive zoom/pan
   - Hover tooltips with image preview
   - Color legend

3. **Info Panel** (right sidebar):
   - Selected point details
   - Model statistics
   - Dataset info

**Callbacks**:
- Update plot on parameter change
- Recalculate UMAP on button click
- Show image on hover
- Update info panel on selection

**Styling**:
- Bootstrap 5 for responsive layout
- Custom CSS for polish
- Dark/light theme toggle

**Run Command**:
```bash
python visualization_app.py --port 8050
```

---

## 5. Differences from MNIST_Diff_Viz

### Enhancements

1. **Multi-step Visualization**
   - Track activations across denoising steps
   - Animate evolution over time
   - Compare different timesteps

2. **Text Conditioning Support**
   - For SDXL/SDv1.5: display prompts
   - Cluster by semantic similarity
   - Filter by keywords

3. **Higher Resolution**
   - SDXL: 1024x1024 (vs 28x28 MNIST)
   - Thumbnail generation for performance
   - Lazy loading for large datasets

4. **Multiple Architectures**
   - Model switcher in UI
   - Consistent interface across models
   - Per-model configuration

5. **Layer Comparison**
   - Side-by-side view of different layers
   - Correlation analysis
   - Layer interpolation visualization

### Technical Considerations

1. **Performance**
   - Precompute UMAP embeddings
   - Use thumbnails for hover previews
   - Implement pagination for large datasets
   - Cache computed results

2. **Storage**
   - Compressed activation storage (NPZ)
   - Thumbnail generation (256x256)
   - Metadata databases (SQLite optional)

3. **Scalability**
   - Support incremental dataset updates
   - Parallel processing for batch generation
   - Distributed UMAP for large datasets

---

## 6. Dependencies

**Core**:
- `dash` - Web framework
- `plotly` - Interactive plots
- `umap-learn` - Dimensionality reduction
- `torch` - Model inference
- `diffusers` - SDXL/SDv1.5 models
- `pandas` - Data handling
- `pillow` - Image processing

**DMD2 Specific**:
- All requirements from `requirements.txt`
- Pre-downloaded model checkpoints

**Optional**:
- `dash-bootstrap-components` - UI styling
- `scikit-learn` - Additional clustering
- `colorcet` - Better color palettes
- `sqlalchemy` - Metadata database

---

## 7. Usage Workflow

### Quick Start
```bash
# 1. Download pretrained model
bash scripts/download_hf_checkpoint.sh "imagenet/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch_fid1.51_checkpoint_model_193500" checkpoints/

# 2. Generate dataset
python generate_dataset.py --model imagenet --checkpoint_path checkpoints/ --num_samples 5000

# 3. Process embeddings
python process_embeddings.py --model imagenet --layer midblock

# 4. Launch visualizer
python visualization_app.py
```

### Advanced Usage
```bash
# Generate SDXL dataset with custom prompts
python generate_dataset.py \
  --model sdxl \
  --checkpoint_path checkpoints/sdxl/ \
  --prompt_file custom_prompts.txt \
  --num_samples 2000 \
  --steps 4

# Process with custom UMAP parameters
python process_embeddings.py \
  --model sdxl \
  --layer "midblock,upblock2" \
  --n_neighbors 50 \
  --min_dist 0.05
```

---

## 8. Future Extensions

1. **Interactive Editing**
   - Click point to edit prompt/class
   - Regenerate with interpolation
   - Real-time updates

2. **Comparison Mode**
   - Compare different models
   - Show evolution across training iterations
   - A/B testing interface

3. **Export Features**
   - Export high-res plots
   - Download selected samples
   - Generate reports

4. **Analysis Tools**
   - Cluster statistics
   - Nearest neighbor search
   - Outlier detection

---

## 9. Resource Requirements

### Memory & Disk Estimates (Detailed in RESOURCE_REQUIREMENTS.md)

**ImageNet (64x64)**:
- Per sample: ~210 KB (192 KB activations + 10 KB image + metadata)
- 1,000 samples: ~203 MB
- 10,000 samples: ~2.0 GB
- GPU memory (batch 64): ~1.5 GB

**SDXL (1024x1024, mid_block only)**:
- Per sample: ~3.5 MB (2.5 MB activations + 1 MB image)
- 1,000 samples: ~3.6 GB
- GPU memory (batch 4): ~8-10 GB

**SDv1.5 (512x512, mid_block only)**:
- Per sample: ~1.2 MB
- 1,000 samples: ~950 MB
- GPU memory (batch 8): ~5-6 GB

### Data Selection Strategy

**Dataset generation** (generate_dataset.py:96-106):
- Default: Random class labels via `np.random.randint(0, 1000, num_samples)`
- Balanced option: `--samples_per_class N` generates N samples per class
- All samples generated sequentially (not pre-selected)
- Controlled by `--seed` for reproducibility

**Visualization** (visualization_app.py):
- Loads ALL samples from dataset (no random selection)
- Can limit with `--max_samples` in process_embeddings.py
- Displays complete dataset in UMAP

**Per-class visualization**:
- Color-coded by class_label for ImageNet
- Can filter/subset in future extensions

## 10. Neighbor Selection Features

### Implementation (ENHANCED - visualizer/visualization_app.py:241-718)

**Features from MNIST_Diff_Viz + Enhancements**:
- ✅ Click point to select (blue highlight)
- ✅ K-nearest neighbors search (sklearn.neighbors.NearestNeighbors)
- ✅ Adjustable k (1-20 via slider)
- ✅ Visual highlighting with differentiation:
  - Selected: Blue circle, size 12
  - KNN neighbors: Red thin ring, width 1
  - Manual neighbors: Red thick ring, width 2
- ✅ Neighbor list display with thumbnails + distances
- ✅ **Toggle selection**: Click to add/remove neighbors
- ✅ **Manual neighbor management**:
  - Click KNN neighbor → removes from KNN list
  - Click manual neighbor → removes from manual list
  - Click other point → adds to manual list
- ✅ **Clear button**: Deselect and clear all neighbors
- ✅ **ImageNet class labels**: Shows "42: goldfish" instead of just "42"

**Interaction Flow**:
1. Click any point on UMAP scatter plot (blue highlight)
2. Adjust k-neighbors slider (default: 5)
3. Click "Find Neighbors" button
4. See neighbor list with 64×64 thumbnails + distances + class names
5. Click any point while selected to toggle manual neighbors
6. Click "✕" to clear everything

**Technical Details**:
- KNN fitted on UMAP coordinates (not raw activations)
- Euclidean distance in 2D UMAP space
- n_neighbors=21 fitted (supports up to k=20)
- Three dcc.Store components for state:
  - `selected-point-store`: Current selection
  - `manual-neighbors-store`: User-added neighbors
  - `neighbor-indices-store`: KNN algorithm results
- Separate scatter traces for selected vs neighbors
- Class labels loaded from `data/imagenet_class_labels.json`

## 10.1 Resume Generation Feature

### Implementation (ADDED - visualizer/generate_dataset.py:104-153)

**Incremental Dataset Building**:
- ✅ Detects existing samples via `metadata/dataset_info.json`
- ✅ Skips already-generated samples
- ✅ Only generates the difference: `new_count = num_samples - existing_count`
- ✅ Maintains seed consistency across batches

**Technical Details**:
- Generates full label sequence for reproducibility
- Slices to skip existing samples
- Merges metadata (existing + new)
- Ensures sample N identical whether generated in batch 1 or batch 10

**Example**:
```bash
# Generate 1000 samples
python generate_dataset.py --num_samples 1000 --seed 10

# Later, add 9000 more (only generates 9000 new)
python generate_dataset.py --num_samples 10000 --seed 10
```

## 11. Next Steps

### Immediate Priorities (COMPLETED ✅)
1. ✅ Implement activation extraction utilities
2. ✅ Create dataset generation pipeline
3. ✅ Build UMAP processing script
4. ✅ Develop Dash application with neighbor selection

### Testing Strategy
1. Start with ImageNet (simpler, smaller resolution)
2. Validate with small dataset (1000 samples)
3. Expand to SDXL once pipeline validated
4. Optimize performance with full datasets

### Next Phase: Testing & Validation
1. End-to-end pipeline test with ImageNet checkpoint
2. Validate neighbor search accuracy
3. Performance profiling (large datasets)
4. SDXL implementation (text-to-image)

---

## References

- **MNIST_Diff_Viz**: https://github.com/mckellcarter/MNIST_Diff_Viz
- **DMD2 Paper**: https://arxiv.org/abs/2405.14867
- **DMD2 Repo**: https://github.com/tianweiy/DMD2
- **Pretrained Models**: https://huggingface.co/tianweiy/DMD2
