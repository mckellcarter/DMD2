# DMD2 Visualizer - Quick Start

## 1-Minute Setup

```bash
# Install dependencies
cd visualizer
pip install -r requirements.txt

# Download checkpoint (ImageNet FID 1.51)
cd ..
bash scripts/download_hf_checkpoint.sh \
  "imagenet/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch_fid1.51_checkpoint_model_193500" \
  checkpoints/

# Return to visualizer
cd visualizer
```

## Generate & Visualize (Small Test)

```bash
# Auto-detect device (CUDA/MPS/CPU)
python generate_dataset.py \
  --model imagenet \
  --checkpoint_path ../checkpoints/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch_fid1.51_checkpoint_model_193500.pth \
  --num_samples 100 \
  --batch_size 50

# Or explicitly use MPS (Apple Silicon)
python generate_dataset.py \
  --model imagenet \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --num_samples 100 \
  --batch_size 32 \
  --device mps

# Process UMAP embeddings (~10 seconds)
python process_embeddings.py --model imagenet

# Launch visualizer (adapter/checkpoint auto-detected from JSON)
./run_visualizer.sh imagenet
```

Open: http://localhost:8050

**Note**: Adapter and checkpoint are auto-detected from the embeddings JSON file.

## Generate & Visualize (Full Dataset)

```bash
# Single GPU/MPS: 5000 samples (~3 minutes on GPU)
python generate_dataset.py \
  --model imagenet \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --num_samples 5000 \
  --batch_size 64 \
  --samples_per_class 5

# Multi-GPU: 10000 samples
accelerate launch generate_dataset_accelerate.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --num_samples 10000 \
  --batch_size 64 \
  --samples_per_class 10

# Process UMAP
python process_embeddings.py \
  --model imagenet \
  --n_neighbors 25 \
  --min_dist 0.1

# Launch (adapter/checkpoint auto-detected)
./run_visualizer.sh imagenet --port 8050
```

## Real ImageNet Extraction (NPZ Format)

Extract activations from real ImageNet64 images (10-100x faster than JPEG):

```bash
# Download ImageNet64 NPZ files first
# Place in: data/Imagenet64_train_npz/

# Quick test: 1000 samples from all classes
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 1000 \
  --batch_size 128 \
  --device mps

# Class-balanced: 100 classes, 50 samples each (5000 total)
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 5000 \
  --num_classes 100 \
  --batch_size 128

# Specific classes: animals (classes 0-9)
python extract_real_imagenet.py \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --npz_dir data/Imagenet64_train_npz \
  --num_samples 1000 \
  --target_classes "0,1,2,3,4,5,6,7,8,9" \
  --batch_size 128

# Process embeddings (propagates adapter/checkpoint to JSON)
python process_embeddings.py \
  --model imagenet_real \
  --n_neighbors 25 \
  --min_dist 0.1

# Launch visualizer (adapter/checkpoint auto-detected)
./run_visualizer.sh imagenet_real
```

**Performance**: ~2000-5000 samples/sec on GPU, ~300-500 on MPS

See `REAL_IMAGENET_GUIDE.md` for JPEG format and full documentation.

## Troubleshooting

**Import errors**: `pip install -r requirements.txt` (from visualizer/)

**CUDA OOM**: Reduce `--batch_size` to 32 or 16

**MPS (Apple Silicon)**:
- Use `--device mps` or let auto-detect
- Reduce `--batch_size` to 32 or 16 for M1/M2
- Falls back to CPU if MPS fails

**Multi-GPU setup**:
- Run `accelerate config` first time
- Use `generate_dataset_accelerate.py`
- Only works with CUDA (not MPS)

**Checkpoint not found**: Check path matches download location

**No embeddings found**: Run `process_embeddings.py` first

**Port in use**: Add `--port 8051` to launch command

## File Locations

- **Images**: `data/images/imagenet/sample_NNNNNN.png`
- **Activations**: `data/activations/imagenet/sample_NNNNNN.npz`
- **Embeddings**: `data/embeddings/imagenet_umap_n15_d0.1.csv`
- **Metadata**: `data/metadata/imagenet/dataset_info.json`

## Next Steps

- Adjust UMAP parameters in UI (recalculate button)
- Export embeddings (export button)
- Generate larger datasets
- Try different checkpoints (FID 1.28 vs 1.51 vs 2.61)

See README.md for complete documentation.
