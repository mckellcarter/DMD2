# Feature Documentation

## Interactive Neighbor Selection

### Overview
Manual and automated neighbor exploration in UMAP space using K-nearest neighbors and click-to-toggle selection.

### Components

**Selection States**:
- `selected-point-store`: Currently selected point index
- `manual-neighbors-store`: User-added neighbor indices
- `neighbor-indices-store`: K-nearest neighbor indices from algorithm

**Visual Indicators**:
- Selected point: Blue circle, size 12
- KNN neighbors: Red ring, width 1
- Manual neighbors: Red ring, width 2

### Usage Flow

1. **Select point**: Click any point → blue highlight
2. **Find KNN**: Click "Find Neighbors" → red thin rings on K-nearest neighbors
3. **Toggle neighbors**: Click any point while selected:
   - If in KNN list → removes from KNN
   - If in manual list → removes from manual
   - If in neither → adds to manual list
4. **Clear all**: Click "✕" button → clears selection, KNN, and manual neighbors

### Implementation Details

**Callback chain** (visualization_app.py):
1. `display_selected` (lines 438-520): Handles click, toggles neighbors, updates stores
2. `update_neighbors` (lines 555-633): Renders neighbor list with class labels
3. `highlight_neighbors` (lines 682-718): Adds visual highlights to plot

**Toggle logic** (lines 489-504):
```python
if point_idx in manual_neighbors:
    manual_neighbors.remove(point_idx)
elif point_idx in knn_neighbors:
    new_knn = [idx for idx in knn_neighbors if idx != point_idx]
else:
    manual_neighbors.append(point_idx)
```

## ImageNet Class Labels

### Overview
Human-readable class names displayed alongside numeric IDs (0-999) in all UI elements.

### Data Source
- File: `data/imagenet_class_labels.json`
- Format: `{"0": ["n01440764", "tench"], ...}`
- Loaded at app startup via `load_class_labels()` (lines 59-76)

### Display Locations

1. **Hover text** (lines 408-414):
   ```python
   customdata = [[f"{r['class_label']}: {self.get_class_name(r['class_label'])}",
                  r['sample_id'], r['image_path']]
                 for _, r in self.df.iterrows()]
   ```

2. **Selected details** (lines 481-484):
   ```python
   class_id = int(sample['class_label'])
   class_name = self.get_class_name(class_id)
   f"Class: {class_id}: {class_name}"
   ```

3. **Neighbor list** (lines 627-630):
   ```python
   class_id = int(neighbor_sample['class_label'])
   class_name = self.get_class_name(class_id)
   html.P(f"Class: {class_id}: {class_name}")
   ```

## Resume Generation

### Overview
Incrementally append to datasets by detecting existing samples and only generating new ones.

### Implementation (generate_dataset.py)

**Detection** (lines 104-122):
```python
if metadata_path.exists():
    with open(metadata_path, 'r') as f:
        existing_data = json.load(f)
        existing_metadata = existing_data.get('samples', [])
        start_sample_idx = len(existing_metadata)
    print(f"Resuming from sample {start_sample_idx}")

    if start_sample_idx >= num_samples:
        print(f"Dataset already has {start_sample_idx} samples")
        return
```

**Seed consistency** (lines 126-144):
- Generate full label sequence to maintain reproducibility
- Slice labels array to skip already-generated samples
- Ensures sample N is identical whether generated in batch 1 or batch 2

**Metadata merging** (lines 151, 215-234):
```python
all_metadata = existing_metadata.copy()
# ... append new samples ...
json.dump({"samples": all_metadata}, f)
```

### Usage

```bash
# Generate 1000 samples
python generate_dataset.py --num_samples 1000 --seed 10

# Later, add 4000 more (only generates 4000 new)
python generate_dataset.py --num_samples 5000 --seed 10
```

**Requirements**:
- Same `--seed` value for reproducibility
- Same checkpoint and parameters
- Metadata file exists at `data/metadata/{model}/dataset_info.json`

## Device Auto-Detection

### Overview
Automatically selects best available device: CUDA > MPS > CPU

### Implementation (device_utils.py)

```python
def get_device(preferred_device: str = None) -> str:
    if preferred_device:
        return preferred_device

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

### Platform Support

**CUDA** (NVIDIA GPUs):
- Full PyTorch support
- Multi-GPU via Accelerate
- Best performance

**MPS** (Apple Silicon):
- M1/M2/M3 support
- Single-device only
- Some ops slower than CUDA
- Automatic fallback to CPU on unsupported ops

**CPU**:
- Universal fallback
- Very slow for generation
- Suitable for small batches only

## UMAP Embedding Processing

### Overview
Compute 2D or 3D UMAP projections from high-dimensional activations.

### Pipeline (process_embeddings.py)

1. **Load activations** (lines 17-69):
   - Read .npz files from activation directory
   - Flatten to 1D vectors per sample
   - Stack into (N, D) matrix

2. **Normalize** (lines 103-106):
   - StandardScaler normalization (zero mean, unit variance)
   - Optional via `--no_normalize` flag

3. **Compute UMAP** (lines 108-122):
   - scikit-learn UMAP implementation
   - Configurable parameters: n_neighbors, min_dist, metric
   - Output: (N, 2) or (N, 3) coordinates

4. **Save results** (lines 140-162):
   - CSV with UMAP coords + metadata
   - JSON with parameters for reproducibility

### Parameters

**n_neighbors** (default: 15):
- Controls local vs global structure
- Low (5-10): Emphasizes local clusters
- High (50-100): Preserves global structure

**min_dist** (default: 0.1):
- Controls point spread
- Low (0.0-0.1): Tight clusters
- High (0.5-1.0): Even distribution

**metric** (default: euclidean):
- Distance function for activation space
- Options: euclidean, cosine, manhattan

## Activation Extraction

### Overview
Hook-based capture of intermediate UNet activations during generation.

### Layer Patterns (extract_activations.py)

**ImageNet** (DhariwalUNet):
- encoder_bottleneck
- midblock
- encoder_block_{0-3}
- decoder_block_{0-3}

**SDXL/SDv1.5** (UNet2DConditionModel):
- down_block_{0-3}
- mid_block
- up_block_{0-3}

### Storage Format

**.npz files** (compressed numpy):
```python
{
    'encoder_bottleneck': ndarray(1, C, H, W),
    'midblock': ndarray(1, C, H, W),
    ...
}
```

**.json metadata**:
```json
{
    "sample_id": "sample_000042",
    "class_label": 42,
    "seed": 10
}
```

### Memory Optimization

- Single forward pass captures multiple layers
- Hooks removed after extraction
- Activations saved immediately, not accumulated
- Per-sample files enable incremental processing
