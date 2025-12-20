# Issue 0000: Large Trajectory Jumps in UMAP Space

## Problem
Generated samples show large jumps in UMAP space during multi-step denoising trajectories. The trajectory path from intended center to final position travels unexpectedly far, sometimes crossing the entire UMAP plot.

## Observed Behavior
- Trajectory starts at "intended" point (center of selected neighbors in UMAP space)
- Step 0, 1, 2, ... positions jump significant distances
- Final generated image may land far from the intended region
- Intermediate images show gradual refinement but UMAP positions don't reflect locality

## Background Context

### How Generation Works
1. User selects neighbors on UMAP plot
2. Calculate 2D center of neighbors
3. UMAP `inverse_transform` to get high-D activation vector
4. Split into per-layer activations, register as masks on model
5. Run multi-step denoising with `generate_with_masked_activation_multistep()`
6. At each step, extract activations and project through UMAP

### Key Files
- `visualization_app.py`: Main app, trajectory visualization (lines 1240-1283)
- `generate_from_activation.py`: `generate_with_masked_activation_multistep()` (lines 171-334)
- `activation_masking.py`: Hook-based activation replacement

### Current Parameters
- `num_steps`: 4 (configurable)
- `mask_steps`: defaults to num_steps (all steps masked)
- `sigma_max`: 80.0
- `sigma_min`: 0.002
- `extract_layers`: sorted list of layers used for UMAP

## Hypotheses to Investigate

### 1. UMAP Projection Artifacts
UMAP is non-linear - small changes in high-D can map to large 2D distances, especially in sparse regions or near manifold boundaries.
- **Test**: Compare raw high-D distances between steps vs UMAP distances
- **Test**: Check if jumps correlate with distance from training data manifold

### 2. Activation Extraction Mismatch
The layers extracted for trajectory projection may not match exactly what's being masked.
- **Check**: Are `extract_layers` and masked layers identical?
- **Check**: Layer order consistency (both use `sorted()`)

### 3. Layer-Specific Behavior
Different layers may have different dynamics during denoising.
- **Test**: Visualize per-layer activation changes separately
- **Test**: UMAP trained on single layer vs concatenated

### 4. Sigma Schedule Effects
High initial sigma (80.0) means early steps have extreme noise levels.
- Early steps: mostly noise, activations may be dominated by noise patterns
- Later steps: signal emerges, activations shift dramatically
- **Test**: Compare trajectories with different sigma schedules

### 5. UMAP Inverse Transform Accuracy
`inverse_transform` is approximate - the reconstructed high-D point may not perfectly represent the 2D center.
- **Test**: Project the inverse-transformed point back through UMAP, compare to original center
- **Metric**: `||umap(inverse_umap(x)) - x||`

### 6. Activation Shape/Normalization
- Activations are scaled by StandardScaler during UMAP training
- Same scaler must be applied before transform
- **Check**: Scaler applied consistently in trajectory projection?

### 7. Batch Dimension Handling
- Generation uses batch size 1
- UMAP was trained on activations from dataset (potentially different batch handling)
- **Check**: Shape consistency `(1, D)` vs `(D,)`

## Diagnostic Steps

1. **Log high-D distances**: Add logging to show Euclidean distance between consecutive step activations in high-D space

2. **Verify inverse transform**: After inverse_transform, immediately forward-transform and log distance from original 2D point

3. **Per-layer analysis**: Project each layer separately through its own UMAP to see which layer(s) cause jumps

4. **Compare to non-masked generation**: Generate without masking, see if trajectory is smoother

5. **Visualize activation histograms**: Plot activation distributions at each step to see if they're shifting dramatically

## Related Code Locations

### Trajectory extraction (generate_from_activation.py:283-300)
```python
if extractor is not None:
    acts = extractor.get_activations()
    layer_acts = []
    for layer_name in sorted(extract_layers):
        act = acts.get(layer_name)
        if act is not None:
            if len(act.shape) == 4:
                B, C, H, W = act.shape
                act = act.reshape(B, -1)
            layer_acts.append(act.numpy())
    if layer_acts:
        concat_act = np.concatenate(layer_acts, axis=1)
        trajectory_activations.append(concat_act)
```

### Trajectory projection (visualization_app.py:1128-1149)
```python
for step_idx, act in enumerate(trajectory_acts):
    if self.umap_scaler is not None:
        act = self.umap_scaler.transform(act)
    coords = self.umap_reducer.transform(act)
    # ... save intermediate image and coords
```

### Inverse transform (visualization_app.py:978-979)
```python
center_activation = self.umap_reducer.inverse_transform(center_2d)
```

## Session Work Completed (Dec 19, 2025)

### Bugs Fixed
1. Trajectory hover not working - `customdata[0]` type mismatch (string vs int)
2. "Intended" point showing ImageNet image - missing customdata on trace
3. Clear button not working - wrong component ID (`umap-plot` vs `umap-scatter`)
4. Clear button causing index errors - added bounds checking

### Features Added
1. Trajectory grid hover - shows all steps with hovered step highlighted
2. Clear button resets selection state

## Session 2 Findings (Dec 19, 2025)

### Root Cause Found: UMAP inverse_transform is Broken

Diagnostic output showed massive round-trip error:
```
[DIAG] Inverse transform round-trip error: 4.0790
[DIAG] Original center: (-2.876, 3.838)
[DIAG] Round-trip center: (1.187, 4.190)
```

The inverse_transform was placing Step 0 at completely wrong location (4+ UMAP units away from intended).

### Solution Implemented: High-D Averaging

Changed architecture to avoid inverse_transform entirely:
- **Old**: Select neighbors in UMAP → average 2D coords → inverse_transform → high-D activation
- **New**: Select neighbors → average directly in high-D activation space → forward transform for visualization only

Key changes:
1. `fit_nearest_neighbors()` - Now fits KNN on high-D activations (scaled)
2. `find_neighbors()` - Queries using high-D activations
3. `generate_from_neighbors()` - Averages neighbor activations directly, forward transforms for visualization

### Results After Fix

Starting point now correct. But Step 0→1 still shows large UMAP jump:
```
[DIAG] Step 0->1: high-D=398.88, UMAP=4.3932
[DIAG] Step 1->2: high-D=451.11, UMAP=2.5251
[DIAG] Step 2->3: high-D=283.74, UMAP=0.6364
```

This is expected with `mask_steps=1` - after first step, model is unconstrained.

### Remaining Investigation

1. **mask_steps effect** - Try `mask_steps=4` (all steps masked) to see if trajectory stays tighter
2. **High-D distances still large** - 280-450 units between steps suggests activations change significantly during denoising
3. **UMAP distortion** - Ratios vary 0.002-0.011, UMAP may amplify certain directions

## Next Steps
1. Test with `mask_steps=num_steps` to keep activations constrained
2. Investigate per-layer activation changes
3. Consider whether trajectory visualization is meaningful given denoising dynamics
