# Session Summary: Generate from Neighbors Feature

## Date: 2025-11-22

## Overview

Added interactive "Generate from Neighbors" feature to DMD2 visualizer. Users can now select neighbors on the UMAP plot and generate new images by interpolating their activations in the latent space.

## New Features

### 1. UMAP Inverse Transform Support

**Modified**: `process_embeddings.py`

- `compute_umap()` now returns `(embeddings, reducer, scaler)` tuple
- `save_embeddings()` saves fitted UMAP model and scaler to `.pkl` file
- Enables inverse_transform to map 2D UMAP coordinates back to high-dimensional activation space

**Key Changes**:
- Added `pickle` import
- UMAP model and StandardScaler now saved alongside embeddings CSV
- Example: `imagenet_umap_n15_d0.1.pkl` saved with `imagenet_umap_n15_d0.1.csv`

### 2. Activation Masking System

**New File**: `activation_masking.py`

Hook-based system for holding layer outputs constant during generation.

**Classes**:
- `ActivationMask`: Manages hooks that replace layer outputs with fixed activations
  - Supports batch expansion (single mask applied to batch)
  - Compatible with both ImageNet (DhariwalUNet) and SDXL/SDv1.5 (UNet2DConditionModel)
  - Context manager support for automatic cleanup

**Key Functions**:
- `set_mask(layer_name, activation)`: Set fixed activation for a layer
- `register_hooks(model, layers)`: Install masking hooks
- `remove_hooks()`: Clean up hooks
- `load_activation_from_npz()`: Load layer activation from saved file
- `unflatten_activation()`: Reshape flattened activation to spatial dimensions

### 3. Generation Module

**New File**: `generate_from_activation.py`

Generates new images with masked activations using DMD2 models.

**Key Functions**:
- `create_imagenet_generator(checkpoint_path, device)`: Load DMD2 generator
- `generate_with_masked_activation(generator, mask, ...)`: Single-step generation with fixed layer activations
- `save_generated_sample(image, activations, metadata, ...)`: Save generated image and activations to dataset
- `infer_activation_shape(generator, layer_name)`: Determine spatial dimensions of layer output

**Generation Pipeline**:
1. Load DMD2 checkpoint
2. Register activation masks on target layers
3. Generate image (noise * sigma → model → image)
4. Save image, activations, and metadata
5. Return generated image for visualization

### 4. Visualizer UI and Logic

**Modified**: `visualization_app.py`

**New UI Components**:
- "Generate from Neighbors" section with:
  - "Generate Image" button (green)
  - Status display area
  - Automatic enable/disable based on checkpoint availability

**New Initialization Parameters**:
- `checkpoint_path`: Path to DMD2 model checkpoint
- `device`: Generation device (cuda/mps/cpu)
- `umap_reducer`: Fitted UMAP model for inverse_transform
- `umap_scaler`: Fitted StandardScaler
- `generator`: Loaded DMD2 model (lazy-loaded on first generation)
- `layer_shapes`: Cache of layer activation shapes

**New Callback**: `generate_from_neighbors()`

**Generation Workflow**:
1. **Validate**: Check selected point and neighbors exist
2. **Calculate center**: Average neighbor coordinates in 2D UMAP space
3. **Inverse transform**: Map center back to high-dimensional activation space
4. **Un-normalize**: Apply inverse StandardScaler transform
5. **Split activations**: Separate concatenated activation back into per-layer tensors
6. **Infer shapes**: Determine spatial dimensions for each layer (cached)
7. **Unflatten**: Reshape flattened activations to (1, C, H, W)
8. **Create mask**: Set activation masks for each layer
9. **Generate**: Single-step DMD2 generation with masked activations
10. **Extract**: Capture actual generated activations via hooks
11. **Save**: Store image, activations, and metadata to dataset
12. **Update plot**: Add new point as green star at neighbor center
13. **Update selection**: Set new point as selected

**Visual Markers**:
- Selected point: Blue circle
- KNN neighbors: Red thin ring
- Manual neighbors: Red thick ring
- **Generated image: Green star** (new!)

### 5. Documentation Updates

**Modified**: `README.md`

- Added `--checkpoint_path` and `--device` to launcher examples
- Documented "Generate from Neighbors" feature in Usage section
- Added step-by-step generation workflow
- Updated Features list

## Technical Implementation Details

### UMAP Inverse Transform

```python
# Forward: activation space → 2D
embeddings = umap_reducer.fit_transform(normalized_activations)

# Backward: 2D → activation space
center_2d = np.mean(neighbor_coords, axis=0)
center_activation = umap_reducer.inverse_transform(center_2d)
center_activation = scaler.inverse_transform(center_activation)  # Un-normalize
```

### Activation Splitting

Activations are concatenated during UMAP processing (in sorted layer order):
```python
# During UMAP: [encoder_bottleneck features | midblock features] → UMAP

# During generation: reverse the process
offset = 0
for layer_name in sorted(layers):
    shape = layer_shapes[layer_name]
    size = np.prod(shape)
    layer_act = center_activation[offset:offset+size]
    activation_dict[layer_name] = unflatten(layer_act, shape)
    offset += size
```

### Hook-based Masking

```python
def _get_masking_hook_fn(self, name: str):
    def hook(module, input, output):
        if name in self.masks:
            # Replace layer output with fixed activation
            masked = self.masks[name].to(output.device, output.dtype)
            # Expand to batch size if needed
            if masked.shape[0] == 1 and output.shape[0] > 1:
                masked = masked.expand(output.shape[0], -1, -1, -1)
            return masked
        return output
    return hook
```

### Single-Step Generation

```python
# DMD2 single-step generation (from demo/imagenet_example.py)
noise = torch.randn(batch_size, 3, 64, 64, device=device)
labels = torch.eye(1000, device=device)[class_indices]
sigma = 80.0  # Conditioning sigma

generated_images = generator(
    noise * sigma,           # Scaled noise
    torch.ones(batch_size) * sigma,  # Timestep
    labels                   # One-hot class labels
)

# With hooks registered, specified layers are replaced with fixed activations
```

## Usage Example

```bash
# 1. Generate dataset with activations
python generate_dataset.py \
  --model imagenet \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --num_samples 1000 \
  --layers encoder_bottleneck,midblock

# 2. Process embeddings (saves UMAP model)
python process_embeddings.py \
  --model imagenet \
  --n_neighbors 15 \
  --min_dist 0.1

# 3. Launch visualizer with generation enabled
python visualization_app.py \
  --embeddings data/embeddings/imagenet_umap_n15_d0.1.csv \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --device cuda

# 4. In the UI:
#    - Click a point to select
#    - Click "Find Neighbors" (or manually click neighbors)
#    - Click "Generate Image"
#    - New image appears as green star at neighbor center
```

## Key Design Decisions

### 1. **Lazy Model Loading**
Generator is loaded only when first generation is requested, not at startup. Saves memory and startup time.

### 2. **Layer Shape Inference**
Layer spatial dimensions are inferred by running a dummy forward pass. Cached for subsequent generations.

### 3. **Batch Expansion in Hooks**
Masks are stored as single-sample tensors (1, C, H, W) and automatically expanded to batch size during forward pass. Allows batch generation with same mask.

### 4. **Center Placement**
Generated images are placed at the exact center of neighbors in UMAP space (not re-projected). This shows the intended interpolation point.

### 5. **Activation Capture**
Both mask activations (set) and generated activations (captured) are saved. Allows verification that masking worked correctly.

## File Structure (Updated)

```
visualizer/
├── Planning/
│   └── SESSION_NEIGHBOR_GENERATION.md  # This file
├── activation_masking.py               # NEW: Hook-based masking
├── generate_from_activation.py         # NEW: Generation module
├── process_embeddings.py               # MODIFIED: Save UMAP model
├── visualization_app.py                # MODIFIED: Generation UI + callback
├── README.md                           # MODIFIED: Documentation
├── extract_activations.py              # (unchanged, used by generation)
├── generate_dataset.py                 # (unchanged)
└── data/
    ├── embeddings/
    │   ├── imagenet_umap_n15_d0.1.csv
    │   ├── imagenet_umap_n15_d0.1.json
    │   └── imagenet_umap_n15_d0.1.pkl  # NEW: UMAP model
    ├── images/imagenet/
    │   └── sample_001000_generated.png # NEW: Generated images
    └── activations/imagenet/
        └── sample_001000_generated.npz # NEW: Generated activations
```

## Dependencies (No New Additions)

All required packages already in `requirements.txt`:
- `torch` - Model inference, hooks
- `umap-learn` - UMAP inverse_transform
- `scikit-learn` - StandardScaler
- `dash` / `plotly` - UI
- `numpy` / `pandas` - Data handling
- `Pillow` - Image I/O

## Limitations & Future Work

### Current Limitations

1. **ImageNet Only**: SDXL/SDv1.5 support not yet implemented
2. **Single Layer Set**: Only layers used during UMAP processing can be masked
3. **No Real-time UMAP**: Generated points use neighbor center, not re-projected activations
4. **Class Label Inheritance**: Generated image uses same class as selected point

### Future Enhancements

1. **Multi-layer Selection**: Allow masking different layer subsets
2. **Dynamic Re-projection**: Re-run UMAP to get true position of generated activations
3. **Batch Generation**: Generate multiple variations from same center
4. **Interactive Editing**: Adjust center point before generation
5. **SDXL/SDv1.5 Support**: Extend to text-to-image models
6. **Activation Interpolation**: Weighted combinations of specific samples
7. **Export Workflow**: Save generation history as reproducible script

## Testing Checklist

- [ ] UMAP model saves correctly (`.pkl` file exists)
- [ ] UMAP inverse_transform produces valid activations
- [ ] Generator loads from checkpoint without errors
- [ ] Layer shapes inferred correctly
- [ ] Activation splitting matches concatenation
- [ ] Hooks register and fire during generation
- [ ] Generated image saves to correct path
- [ ] New point appears on UMAP plot as green star
- [ ] New point can be selected and used for further generation
- [ ] Multiple generations work without reloading model

## Status

✅ **Implementation Complete**

All planned features implemented. Ready for end-to-end testing with real ImageNet dataset.

## Next Steps

1. **Test with existing dataset**: Run full pipeline with 1000-sample ImageNet dataset
2. **Verify generation quality**: Check that generated images are reasonable
3. **Test edge cases**: Single neighbor, many neighbors, repeated generation
4. **Performance profiling**: Measure generation time, memory usage
5. **User testing**: Get feedback on UI/UX
6. **Consider SDXL**: Plan extension to text-to-image models
