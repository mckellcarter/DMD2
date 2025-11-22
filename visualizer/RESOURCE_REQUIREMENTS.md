# DMD2 Visualizer - Resource Requirements

## Memory & Disk Space Estimates

### ImageNet (64x64, DhariwalUNet)

#### Per-Sample Breakdown

**Activations (2 layers: encoder_bottleneck + midblock)**:
- Encoder bottleneck: 768 channels × 8×8 = 49,152 features
- Midblock: 768 channels × 8×8 = 49,152 features
- Total features: 98,304 (float32 = 4 bytes each)
- **Raw size**: ~384 KB/sample
- **Compressed (NPZ)**: ~192 KB/sample

**Generated Images**:
- Resolution: 64×64×3
- **PNG compressed**: ~5-15 KB/sample
- **Uncompressed**: 12 KB/sample

**Metadata**:
- JSON sidecar: ~0.5 KB/sample

**Total per sample**: ~210 KB (compressed)

#### Dataset Size Estimates

| Samples | Activations | Images | Metadata | Total Disk |
|---------|-------------|--------|----------|------------|
| 100 | 19 MB | 1 MB | 0.05 MB | **~20 MB** |
| 1,000 | 192 MB | 10 MB | 0.5 MB | **~203 MB** |
| 5,000 | 960 MB | 50 MB | 2.5 MB | **~1.0 GB** |
| 10,000 | 1.9 GB | 100 MB | 5 MB | **~2.0 GB** |
| 50,000 | 9.6 GB | 500 MB | 25 MB | **~10.1 GB** |

#### Runtime Memory (Generation)

**Batch size 64**:
- Model weights: ~400 MB (DhariwalUNet)
- Batch activations: 64 × 384 KB = 24 MB
- Batch images: 64 × 12 KB = 768 KB
- Gradients: 0 (inference only)
- **Total GPU memory**: ~1.5 GB

**Batch size 128**:
- **Total GPU memory**: ~2.5 GB

**UMAP Processing** (CPU):
- Load all activations into RAM
- 10,000 samples: ~2 GB RAM
- 50,000 samples: ~10 GB RAM
- UMAP computation: +1-2 GB overhead

---

### SDXL (1024×1024, UNet2DConditionModel)

#### Per-Sample Breakdown

**Activations (mid_block only)**:
- Mid block: 1280 channels × 32×32 = 1,310,720 features
- **Raw size**: ~5 MB/sample
- **Compressed (NPZ)**: ~2.5 MB/sample

**Activations (all 7 layers: 3 down + mid + 3 up)**:
- Total: 75 MB/sample raw
- **Compressed (NPZ)**: ~37.5 MB/sample

**Generated Images**:
- Resolution: 1024×1024×3
- **PNG compressed**: ~500 KB - 2 MB/sample
- **Thumbnail (256×256)**: ~50 KB/sample

**Text Prompts**:
- Average prompt: ~50 bytes
- CLIP embeddings (cached): 77 tokens × 768 dims × 4 bytes = 237 KB

**Total per sample** (mid_block only): ~3.5 MB
**Total per sample** (all layers): ~40 MB

#### Dataset Size Estimates (mid_block only)

| Samples | Activations | Images | Prompts | Total Disk |
|---------|-------------|--------|---------|------------|
| 100 | 250 MB | 100 MB | 0.01 MB | **~350 MB** |
| 1,000 | 2.5 GB | 1 GB | 0.1 MB | **~3.6 GB** |
| 5,000 | 12.5 GB | 5 GB | 0.5 MB | **~17.5 GB** |
| 10,000 | 25 GB | 10 GB | 1 MB | **~35 GB** |

**Note**: All-layer extraction multiplies activation storage by ~10-15×

#### Runtime Memory (Generation)

**Batch size 4**:
- Model weights: ~5 GB (SDXL UNet + VAE + text encoders)
- Batch activations (mid_block): 4 × 5 MB = 20 MB
- Batch images (latent): 4 × 256 MB = 1 GB
- Text encoder cache: ~500 MB
- **Total GPU memory**: ~8-10 GB

**Batch size 8**:
- **Total GPU memory**: ~12-14 GB

**UMAP Processing** (CPU):
- 1,000 samples: ~3 GB RAM
- 5,000 samples: ~15 GB RAM
- 10,000 samples: ~30 GB RAM

---

### SDv1.5 (512×512, UNet2DConditionModel)

#### Per-Sample Breakdown

**Activations (mid_block only)**:
- Mid block: 1280 channels × 16×16 = 327,680 features
- **Raw size**: ~1.3 MB/sample
- **Compressed (NPZ)**: ~650 KB/sample

**Generated Images**:
- Resolution: 512×512×3
- **PNG compressed**: ~200-500 KB/sample

**Total per sample**: ~1.2 MB (mid_block only)

#### Dataset Size Estimates (mid_block only)

| Samples | Activations | Images | Total Disk |
|---------|-------------|--------|------------|
| 100 | 65 MB | 30 MB | **~95 MB** |
| 1,000 | 650 MB | 300 MB | **~950 MB** |
| 5,000 | 3.25 GB | 1.5 GB | **~4.8 GB** |
| 10,000 | 6.5 GB | 3 GB | **~9.5 GB** |

#### Runtime Memory (Generation)

**Batch size 8**:
- Model weights: ~3.5 GB (SDv1.5 UNet + VAE + text encoder)
- Batch activations: 8 × 1.3 MB = 10 MB
- Batch images (latent): 8 × 64 MB = 512 MB
- **Total GPU memory**: ~5-6 GB

---

## Recommendations

### Small Test (Quick validation)
- **ImageNet**: 100-500 samples, batch_size=64
- **Disk**: <100 MB
- **GPU**: 2 GB
- **Time**: <1 minute

### Medium Dataset (Good visualization)
- **ImageNet**: 5,000-10,000 samples, batch_size=64
- **Disk**: 1-2 GB
- **GPU**: 2 GB
- **Time**: 3-5 minutes

### Large Dataset (Publication quality)
- **ImageNet**: 50,000 samples (50/class), batch_size=128
- **Disk**: ~10 GB
- **GPU**: 3 GB
- **Time**: ~15 minutes
- **UMAP**: ~10-20 minutes on CPU

### SDXL Considerations
- Start with 500-1,000 samples due to size
- Use mid_block only initially
- Batch size 4-8 depending on GPU
- Consider thumbnail-only storage for large datasets

---

## Disk Space Planning

### Recommended Structure

```
data/
├── images/           # ~10-20% of total
├── activations/      # ~80-90% of total
├── embeddings/       # <1% (CSV files)
└── metadata/         # <1% (JSON files)
```

### Clean-up Strategy

**After UMAP processing**, can optionally delete:
- Raw activations (keep embeddings CSV only)
- Full-size images (keep thumbnails)

**Space savings**: 90%+ reduction
**Trade-off**: Can't recalculate UMAP with different parameters

---

## Memory Optimization Tips

### Generation Phase
1. Reduce `--batch_size` if OOM
2. Generate in chunks (multiple runs with different seeds)
3. Use `--layers` to limit extraction (e.g., only mid_block)

### UMAP Phase
1. Use `--max_samples` to limit dataset size
2. Run on machine with sufficient RAM (2× activation size)
3. Save embeddings CSV, delete activations afterward

### Visualization Phase
1. Thumbnail images on load (256×256)
2. Base64 encode on-demand (not preloaded)
3. Lazy load image previews
4. Use precomputed embeddings (don't dynamic recalculate for large datasets)

---

## Hardware Requirements

### Minimum (ImageNet small test)
- GPU: 4 GB VRAM
- RAM: 8 GB
- Disk: 1 GB free

### Recommended (ImageNet full dataset)
- GPU: 8 GB VRAM (RTX 3070+)
- RAM: 16 GB
- Disk: 20 GB free

### SDXL (medium dataset)
- GPU: 16 GB VRAM (RTX 4080+, A100)
- RAM: 32 GB
- Disk: 50 GB free

---

## Compression Ratios

**NPZ compression** (numpy `savez_compressed`):
- Typical: 40-60% size reduction
- Activations often compress well (spatial redundancy)
- Can use `np.float16` for additional 50% reduction (with precision loss)

**PNG compression**:
- ImageNet (64×64): 5-15 KB vs 12 KB uncompressed (~60% reduction)
- SDXL (1024×1024): 500 KB - 2 MB vs 3 MB uncompressed (~50% reduction)

**Future**: Consider switching to HDF5 for large datasets (better random access)
