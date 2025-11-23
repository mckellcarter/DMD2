# Session Summary: Real ImageNet Activation Extraction

## Date: 2025-11-23

## Overview

Added batch activation extraction for **real ImageNet images** to enable comparison between real and generated image distributions. This implements the "Real ImageNet Embedding Space" enhancement proposed in FUTURE_ENHANCEMENTS.md.

## New Features

### 1. Real ImageNet Activation Extractor

**New File**: `extract_real_imagenet.py`

Processes real ImageNet images through the DMD2 generator to extract activations at specified layers.

**Key Functions**:
- `preprocess_imagenet_image()`: Load and normalize ImageNet images to match DMD2 format
- `parse_imagenet_path()`: Extract synset ID from ImageNet directory structure
- `extract_real_imagenet_activations()`: Main batch processing function
- `load_imagenet_model()`: Load DMD2 checkpoint
- `get_imagenet_config()`: Get model configuration

**Processing Pipeline**:
1. Load DMD2 checkpoint with activation hooks
2. Discover ImageNet images from directory structure
3. Process images in batches:
   - Load and preprocess images (resize to 64x64, normalize to [-1, 1])
   - Parse synset IDs from directory names
   - Map synset IDs to class IDs and names
   - Run forward pass through generator (with sigma=80.0 conditioning)
   - Extract activations via hooks
   - Save batch activations and metadata
4. Save global dataset metadata

### 2. Batch Metadata Format

Each batch includes full ImageNet identifiers:

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
      "original_path": "/path/to/imagenet/val/n01440764/ILSVRC2012_val_00000001.JPEG"
    },
    ...
  ]
}
```

**Fields**:
- `batch_index`: Position within batch (0-indexed)
- `class_id`: ImageNet class ID (0-999)
- `synset_id`: WordNet synset identifier (e.g., "n01440764")
- `class_name`: Human-readable class name (e.g., "tench")
- `original_path`: Path to original ImageNet image

### 3. Output Structure

```
data/
├── images/imagenet_real/
│   └── sample_000000.JPEG  # Copied from original ImageNet
├── activations/imagenet_real/
│   ├── batch_000000.npz    # Compressed numpy arrays (B, C*H*W)
│   └── batch_000000.json   # Batch metadata with ImageNet IDs
└── metadata/imagenet_real/
    └── dataset_info.json   # Global dataset info
```

### 4. Test Suite

**New File**: `test_extract_real_imagenet.py`

Comprehensive tests for:
- Image preprocessing (resize, normalization, tensor conversion)
- Path parsing (synset ID extraction)
- Batch processing (image discovery, batching)
- Synset mapping (synset → class ID/name)
- Activation format (batch storage, single sample extraction)

**Test Coverage**:
- ✅ Image preprocessing produces correct shape and range
- ✅ Different input sizes handled correctly
- ✅ ImageNet path structures parsed correctly (val/train)
- ✅ Synset to class mapping works
- ✅ Batch metadata has correct structure
- ✅ Activation batches stored in correct format
- ✅ Single samples can be extracted from batches

All 10 tests pass.

### 5. Documentation

**Updated**: `README.md`

Added "Real ImageNet Activation Extraction" section with:
- Usage examples
- Expected ImageNet directory structure
- Output file structure
- Batch metadata format
- Use cases for real vs generated comparison

## Usage Example

```bash
# Extract activations from 1000 ImageNet validation images
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --imagenet_dir /path/to/imagenet \
  --num_samples 1000 \
  --batch_size 64 \
  --layers encoder_bottleneck,midblock \
  --split val \
  --device cuda
```

## Technical Implementation Details

### Image Preprocessing

Real ImageNet images are preprocessed to match DMD2 generated image format:

```python
# Load and resize
img = Image.open(path).convert('RGB')
img = img.resize((64, 64), Image.LANCZOS)

# Normalize to [-1, 1] (same as generated images)
img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
img_tensor = (img_tensor / 127.5) - 1.0  # [0, 255] -> [-1, 1]
```

### Synset Mapping

ImageNet directory structure encodes synset IDs:
```
imagenet/val/n01440764/ILSVRC2012_val_*.JPEG
             ^^^^^^^^^^
             synset ID
```

Synset IDs are mapped to class IDs and names using `imagenet_class_labels.json`:
```python
class_labels_map = {
    "0": ["n01440764", "tench"],
    "1": ["n01443537", "goldfish"],
    ...
}

# Reverse mapping
synset_to_class = {
    "n01440764": (0, "tench"),
    "n01443537": (1, "goldfish"),
    ...
}
```

### Activation Extraction

Activations are extracted using the same DMD2 generator and hooks as generated images:

```python
# Run forward pass with conditioning
sigma = torch.ones(batch_size, device=device) * 80.0
_ = generator(
    images * 80.0,          # Scaled input
    sigma,                   # Timestep
    one_hot_labels           # Class conditioning
)

# Activations captured via hooks
activations = extractor.get_activations()
```

**Note**: Real images are processed through the generator (not just the encoder) to ensure activations are extracted from the same computational graph as generated images.

### Batch Storage Format

Same format as generated images:
- NPZ files store activations with flattened spatial dims: `(B, C*H*W)`
- JSON files store per-sample metadata
- Individual images copied to output directory
- Global metadata tracks all samples

## Use Cases

### 1. Distribution Comparison

Compare activation distributions between real and generated:
```python
# Load real activations
real_acts, real_meta = load_batch_activations("data/activations/imagenet_real/batch_000000")

# Load generated activations
gen_acts, gen_meta = load_batch_activations("data/activations/imagenet/batch_000000")

# Compare distributions
plot_activation_histograms(real_acts, gen_acts)
```

### 2. UMAP on Real Space

Fit UMAP on real ImageNet, project generated samples:
```python
# Fit UMAP on real activations
reducer = UMAP(n_neighbors=15, min_dist=0.1)
real_embeddings = reducer.fit_transform(real_activations)

# Project generated activations into real space
gen_embeddings = reducer.transform(gen_activations)

# Visualize both
plt.scatter(real_embeddings[:, 0], real_embeddings[:, 1], alpha=0.3, label='Real')
plt.scatter(gen_embeddings[:, 0], gen_embeddings[:, 1], alpha=0.8, label='Generated')
```

### 3. Mode Coverage Analysis

Identify which real modes are covered by generator:
```python
# For each real sample, find nearest generated sample
from scipy.spatial import KDTree
tree = KDTree(gen_embeddings)
distances, indices = tree.query(real_embeddings)

# Analyze coverage
coverage = (distances < threshold).mean()
print(f"Mode coverage: {coverage:.2%}")
```

### 4. Class Fidelity Analysis

Check if generated samples land near their class in real space:
```python
# Group by class
for class_id in range(1000):
    real_class = real_embeddings[real_labels == class_id]
    gen_class = gen_embeddings[gen_labels == class_id]

    # Compute centroid distance
    real_centroid = real_class.mean(axis=0)
    gen_centroid = gen_class.mean(axis=0)

    print(f"Class {class_id}: distance = {np.linalg.norm(gen_centroid - real_centroid)}")
```

## File Structure (Updated)

```
visualizer/
├── Planning/
│   ├── SESSION_NEIGHBOR_GENERATION.md
│   ├── FUTURE_ENHANCEMENTS.md
│   └── SESSION_REAL_IMAGENET.md       # NEW: This file
├── extract_activations.py              # (unchanged, shared by both)
├── generate_dataset.py                 # (unchanged, for generated)
├── extract_real_imagenet.py            # NEW: Real ImageNet processing
├── test_extract_real_imagenet.py       # NEW: Tests
├── README.md                           # MODIFIED: Added docs
└── data/
    ├── imagenet_class_labels.json      # (used for synset mapping)
    ├── images/
    │   ├── imagenet/                   # Generated images
    │   └── imagenet_real/              # NEW: Real ImageNet images
    ├── activations/
    │   ├── imagenet/                   # Generated activations
    │   └── imagenet_real/              # NEW: Real ImageNet activations
    └── metadata/
        ├── imagenet/
        └── imagenet_real/              # NEW: Real ImageNet metadata
```

## Key Design Decisions

### 1. **Same Model for Extraction**
Use the DMD2 generator to extract activations from real images (not a separate classifier). This ensures activations are from the same feature space.

### 2. **Batch Format Consistency**
Real ImageNet uses the same batch storage format as generated images for easy comparison.

### 3. **Full ImageNet Identifiers**
Store class_id, synset_id, AND class_name for maximum flexibility in analysis.

### 4. **Original Path Tracking**
Keep reference to original ImageNet file for debugging and verification.

### 5. **Flexible Splits**
Support both validation and training sets via `--split` argument.

## Limitations & Future Work

### Current Limitations

1. **ImageNet-64 Only**: Hardcoded for 64x64 resolution
2. **No Preprocessing Options**: Fixed LANCZOS resize, no augmentation
3. **Memory Constraints**: Loads full batches into memory
4. **No Incremental Processing**: Can't resume interrupted processing

### Future Enhancements

1. **Multi-Resolution Support**: Handle ImageNet-128, ImageNet-256
2. **Preprocessing Variants**: Test different resize methods, augmentations
3. **Streaming Processing**: Process without loading full batches
4. **Resume Capability**: Add checkpoint/resume like `generate_dataset.py`
5. **Dual Visualization**: Extend visualizer to show real + generated together
6. **Distribution Metrics**: Add built-in FID, IS, Precision/Recall computation
7. **Subset Selection**: Support class-balanced sampling, specific class selection

## Testing Checklist

- ✅ Image preprocessing works correctly
- ✅ Synset parsing works for val/train splits
- ✅ Synset to class mapping is accurate
- ✅ Batch metadata has correct structure
- ✅ Activations saved in correct format
- ✅ Tests pass (10/10)
- ✅ Documentation updated
- ⬜ End-to-end test with real ImageNet dataset (requires data)
- ⬜ Integration with UMAP processing (requires real data)
- ⬜ Dual visualization (real + generated) - future work

## Status

✅ **Implementation Complete**

Core functionality implemented and tested. Ready for use with real ImageNet dataset.

## Next Steps

1. **Test with real data**: Run on actual ImageNet validation set
2. **UMAP integration**: Extend `process_embeddings.py` to handle real + generated
3. **Dual visualization**: Update `visualization_app.py` to show both datasets
4. **Analysis tools**: Create scripts for distribution comparison
5. **Documentation**: Add tutorial notebook for real vs generated analysis
