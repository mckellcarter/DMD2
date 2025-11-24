# Future Enhancements - DMD2 Visualizer

## Label Conditioning Options

### Current Behavior
- Generated images use the **selected point's class label** for conditioning
- One-hot encoding: `[0, 0, ..., 1, ..., 0]` (1000-dim vector, single 1)
- Strong semantic control from single class

### Proposed Options

#### 1. Uniform Conditioning
**Concept**: Condition equally across all 1000 ImageNet classes
```python
uniform_labels = torch.ones((num_samples, 1000), device=device) / 1000.0
```

**Expected behavior**:
- May produce averaged/blended features across classes
- Weaker semantic constraints
- Untested - model not trained with uniform conditioning

**Use case**: Explore activation space without class bias

#### 2. Zero/Null Conditioning
**Concept**: No class conditioning
```python
zero_labels = torch.zeros((num_samples, 1000), device=device)
```

**Expected behavior**:
- Unconditional generation based purely on activation masking
- Model may ignore or produce unpredictable results
- Depends on how model handles zero vectors

**Use case**: Pure activation-driven generation

#### 3. Multi-Class Blend
**Concept**: Weight multiple classes (e.g., neighbor classes)
```python
# Average one-hot vectors of neighbor classes
neighbor_labels = [label_1, label_2, ..., label_k]
blended = sum(one_hot(l) for l in neighbor_labels) / k
```

**Expected behavior**:
- Semantic interpolation between classes
- More natural for neighbors from different classes

**Use case**: Cross-class exploration

### Implementation Considerations
- Add dropdown in UI: "Conditioning Mode"
  - Selected Point Class (current)
  - Uniform
  - Zero
  - Neighbor Average
- Update `generate_from_activation.py` to accept conditioning mode
- Test empirically - DMD2 behavior with non-standard conditioning unknown

## Real ImageNet Embedding Space

### Status: ✅ Partially Implemented

### Concept
Define UMAP embedding using **original ImageNet images**, then project **generated images** into that space.

### Current Behavior
- UMAP fitted on generated DMD2 activations
- Embedding space represents generator's internal distribution
- No reference to real ImageNet data

### ✅ Implemented Features (Nov 2025)

#### 1. Extract Real ImageNet Activations ✅
**Script**: `extract_real_imagenet.py`

Supports two input formats:
- **NPZ format** (ImageNet64, recommended): 10-100x faster
- **JPEG format** (original ImageNet): Full resolution support

```bash
# NPZ extraction with class-balanced sampling
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_fid1.51.pth \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 10000 \
  --num_classes 100 \
  --batch_size 128
```

**Features**:
- Class-balanced sampling (`--num_classes`, `--target_classes`)
- Batch NPZ format storage
- Metadata tracking (class labels, synset IDs)
- Saves original images and reconstructions

#### 2. Fit UMAP on Real Data ✅
**Script**: `process_embeddings.py --model imagenet_real`

```bash
python process_embeddings.py \
  --model imagenet_real \
  --n_neighbors 25 \
  --min_dist 0.1
```

**Output**: `data/embeddings/imagenet_real_umap_*.csv`

#### 3. Project Generated Samples ⏳
**Status**: Not yet implemented

Planned implementation:
```python
# Transform generated activations using fitted UMAP
gen_activations = load_generated_activations()
gen_embeddings = reducer.transform(gen_activations)  # Project into real space
```

**Required**:
- Save UMAP reducer object during real data processing
- Load reducer and transform generated samples
- Update visualizer to handle dual datasets

### Visualization Benefits

**Dual-layer visualization**:
- Base layer: Real ImageNet images (gray/background)
- Overlay: Generated images (colored/highlighted)
- Trajectories: Show how generation moves through real space

**Analysis opportunities**:
- Mode coverage: Which real modes does generator cover?
- Mode collapse: Are generated samples clustered tighter than real?
- Distribution shift: How far do generated samples deviate?
- Class fidelity: Do generated "goldfish" land near real goldfish?

### Implementation Steps

1. **Data pipeline**:
   - Download ImageNet validation set
   - Extract activations from real images (same layers)
   - Store in compatible format

2. **UMAP workflow**:
   - Fit on real activations
   - Save fitted reducer (with `pickle`)
   - Transform generated activations

3. **Visualization updates**:
   - Toggle: Show real/generated/both
   - Color scheme: Real (gray), Generated (by class)
   - Interactivity: Click real image to see nearest generated

4. **Metrics**:
   - Coverage: % of real space occupied by generated
   - Precision: Generated samples near real clusters
   - Recall: Real clusters represented by generated

### Technical Considerations

**Computational cost** (measured on real hardware):
- **NPZ format** (ImageNet64):
  - CUDA (V100): ~2000-5000 samples/sec
  - MPS (M2 Ultra): ~300-500 samples/sec
  - 10k samples: ~2-30 seconds
- **JPEG format**:
  - CUDA (V100): ~500 samples/sec
  - 10k samples: ~20 seconds
- UMAP fit: ~5-10 min on 50k samples
- Storage: Batch NPZ format ~1-2 GB for 10k samples

**Activation extraction** (implemented):
- Uses **same model checkpoint** as generation
- Same sigma conditioning (80.0)
- Same layers (encoder_bottleneck, midblock)
- Forward pass through generator, not classifier
- See `extract_real_imagenet.py` and `REAL_IMAGENET_GUIDE.md`

**Alternative**: Use pretrained classifier activations instead of generator activations for comparison

### Use Cases
- **Research**: Analyze DMD2 distribution vs real ImageNet
- **Debugging**: Identify underrepresented modes
- **Quality assessment**: Quantify generation fidelity
- **Interpolation**: Generate samples to "fill gaps" in real distribution

## Other Future Ideas
- TBD
