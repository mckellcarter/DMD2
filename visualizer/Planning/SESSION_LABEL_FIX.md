# Session: Label Bug Fix and Class-Balanced Sampling

**Date:** November 23, 2025
**Branch:** `feature/imagenet-npz-support`

## Overview

Fixed critical label mismapping bug in ImageNet real extraction and added class-balanced sampling feature.

## Issues Discovered

### 1. Visualizer Display Bug
**Problem:** Visualizer showed incorrect class names for real ImageNet samples
**Cause:** `visualization_app.py` was using `get_class_name()` to look up names from `imagenet_class_labels.json` instead of using the `class_name` field from the CSV
**Impact:** Correct labels existed in CSV but were being overridden with looked-up values

### 2. Critical Label Mismapping Bug
**Problem:** ALL labels in extracted ImageNet real data were systematically wrong
**Example:**
- Sample showing kangaroo labeled as "tench" (fish)
- Sample showing bird labeled as "gar" (fish)
- 13 samples with class_label=0, all incorrect (should have been 612, 838, 753, 23, etc.)

**Root Cause:** NPZ files sorted alphabetically instead of numerically
```python
# WRONG (alphabetical):
npz_files = sorted(list(npz_dir.glob('*.npz')))
# Result: batch_1, batch_10, batch_2, batch_3...

# CORRECT (numerical):
npz_files = sorted(
    list(npz_dir.glob('*.npz')),
    key=lambda p: int(p.stem.split('_')[-1])
)
# Result: batch_1, batch_2, batch_3... batch_10
```

**Impact:** Entire extracted dataset had misaligned labels
- File index 1 loaded `train_data_batch_10.npz` instead of `train_data_batch_2.npz`
- All subsequent file indices shifted, scrambling label-to-image mappings
- 999 unique classes detected (missing 1) instead of expected 1000

## Fixes Implemented

### 1. visualization_app.py (4 locations)
**Lines:** 472, 562, 643, 758
**Fix:** Use `class_name` from CSV when available
```python
# Before:
class_name = self.get_class_name(class_id)

# After:
class_name = sample.get('class_name', self.get_class_name(class_id))
```

### 2. extract_real_imagenet.py
**Line:** 202-205
**Fix:** Sort NPZ files numerically
```python
npz_files = sorted(
    list(npz_dir.glob('*.npz')),
    key=lambda p: int(p.stem.split('_')[-1])
)
```

## New Feature: Class-Balanced Sampling

### Motivation
Enable controlled sampling from specific ImageNet classes for focused analysis and balanced datasets.

### Implementation

**New Arguments:**
- `--num_classes`: Number of classes to sample from (default: 1000)
- `--target_classes`: Comma-separated class IDs (e.g., "0,1,2,3")

**Algorithm:**
1. Select target classes (random if not specified)
2. Calculate `samples_per_class = num_samples // num_classes`
3. Iterate through NPZ files sequentially
4. Collect samples until each class has ~`samples_per_class`
5. Stop when target reached

**Benefits:**
- Balanced class distribution
- Efficient (stops early once quota met)
- Reproducible with seed
- Supports subset selection

### Example Usage

```bash
# 100 classes, 50 samples each (5000 total)
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_fid1.51.pth \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 5000 \
  --num_classes 100 \
  --batch_size 128

# Specific classes (animals)
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_fid1.51.pth \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 1000 \
  --target_classes "0,1,2,3,4,5,6,7,8,9"
```

## Testing

### Validation Steps
1. ✓ Verified NPZ sorting with debug script
2. ✓ Confirmed label 753 should map to global index 174793
3. ✓ Identified 13 mislabeled samples (all had class_id=0)
4. ✓ Checked actual labels in NPZ files vs saved labels
5. ✓ Python syntax validation passed

### Before Re-extraction
- Old dataset: 999 unique classes, systematic label misalignment
- 13 samples with class_id=0 (all incorrect)

### After Re-extraction (TODO)
- Should have: 1000 unique classes (or num_classes specified)
- Balanced distribution: min/max/mean samples per class
- Labels match images

## Files Modified

1. `extract_real_imagenet.py`
   - Line 202-205: Numerical NPZ sorting
   - Lines 131-144: Updated function signature
   - Lines 226-275: Class-balanced sampling logic
   - Lines 647-658: New CLI arguments

2. `visualization_app.py`
   - Lines 472, 562, 643, 758: Use CSV class_name

3. Documentation (this session)

## Impact

**Critical:**
- Old extracted data is INVALID and must be re-extracted
- All embeddings/visualizations from old data are incorrect

**Positive:**
- Future extractions will have correct labels
- Class-balanced sampling enables focused research

## Next Steps

1. Delete old extracted data:
   ```bash
   rm -rf data/images/imagenet_real*
   rm -rf data/activations/imagenet_real
   rm -rf data/metadata/imagenet_real
   rm -rf data/embeddings/imagenet_real*
   ```

2. Re-extract with fixed code:
   ```bash
   python extract_real_imagenet.py \
     --checkpoint_path ../checkpoints/imagenet_fid1.51.pth \
     --npz_dir data/Imagenet64_train_npz \
     --num_samples 10000 \
     --num_classes 100 \
     --batch_size 128 \
     --device mps
   ```

3. Re-process embeddings:
   ```bash
   python process_embeddings.py \
     --model imagenet_real \
     --n_neighbors 25 \
     --min_dist 0.1
   ```

4. Verify in visualizer that labels match images

## Lessons Learned

1. **Alphabetical vs Numerical Sorting:** Always be explicit about sort order for numbered files
2. **Data Validation:** Should have verified labels matched images earlier
3. **Class Distribution:** Class-balanced sampling improves dataset quality for analysis
4. **Testing:** Need unit tests for NPZ file ordering and label extraction

## References

- Previous session: `SESSION_NPZ_SUPPORT.md`
- NPZ docs: `NPZ_EXTRACTION_GUIDE.md`
- Real ImageNet docs: `REAL_IMAGENET_GUIDE.md`
