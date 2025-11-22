# Changelog

## Recent Updates

### Neighbor Selection & Interaction (2025-11-22)
- **Toggle neighbor selection**: Click points to add/remove from neighbor list
- **Clear button**: Deselect and clear all neighbors at once
- **Separate highlighting**:
  - Selected point: blue, size 12
  - KNN neighbors: red ring, width 1
  - Manual neighbors: red ring, width 2
- **ImageNet class labels**: Human-readable names in hover, selection, and neighbor displays
- **Improved neighbor list**: KNN neighbors first, manual additions at bottom

### Resume Generation (2025-11-22)
- **Incremental dataset generation**: Append to existing datasets without regenerating
- **Automatic detection**: Checks metadata to skip already-generated samples
- **Seed consistency**: Maintains reproducible label sequence across batches
- **Example**:
  ```bash
  # Generate first 1000 samples
  python generate_dataset.py --model imagenet --checkpoint_path ckpt.pth --num_samples 1000

  # Later, add 9000 more (only generates 9000 new samples)
  python generate_dataset.py --model imagenet --checkpoint_path ckpt.pth --num_samples 10000
  ```

### Bug Fixes (2025-11-22)
- Fixed batch size mismatch with `samples_per_class` parameter
- Fixed samples_per_class cycling to support arbitrary num_samples
- Fixed legend accumulation on neighbor clicks
- Fixed neighbor highlighting visibility

## Features

### Core Functionality
- Multi-model support (ImageNet, SDXL, SDv1.5)
- Interactive UMAP visualization
- Layer-specific activation extraction
- K-nearest neighbors search
- Manual neighbor selection
- Hover image previews
- Class label display

### Device Support
- Auto-detection (CUDA > MPS > CPU)
- Apple Silicon MPS optimization
- Multi-GPU via Accelerate
- Fallback handling

### Data Management
- Incremental dataset generation
- Compressed activation storage (.npz)
- Metadata tracking (JSON)
- UMAP parameter persistence
- Resume capability

## Known Issues

### Not Yet Implemented
- SDXL dataset generation
- SDv1.5 dataset generation
- 3D UMAP visualization
- Export selected neighbors
- Batch neighbor analysis

### Platform Limitations
- MPS doesn't support multi-device
- Some MPS ops slower than CUDA
- Large datasets (>10k) slow UMAP recalculation
