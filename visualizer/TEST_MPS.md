# Testing on MPS (Apple Silicon)

Quick guide for testing the visualizer on M1/M2/M3 Macs.

## Quick Test (100 samples, ~2 minutes)

```bash
cd visualizer

# Check MPS is available
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Generate small dataset
python generate_dataset.py \
  --model imagenet \
  --checkpoint_path ../checkpoints/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch_fid1.51_checkpoint_model_193500.pth \
  --num_samples 100 \
  --batch_size 24 \
  --layers encoder_bottleneck,midblock \
  --device mps

# Process UMAP
python process_embeddings.py --model imagenet

# Launch visualizer
./run_visualizer.sh imagenet
```

## Expected Output

```
Device: mps (Apple Silicon MPS)
Loading ImageNet model from ../checkpoints/...
Generating batches: 100%|████████| 5/5 [01:45<00:00, 21.0s/it]
Generated 100 samples
```

## Device Info

The code will print:
- Device: mps (Apple Silicon MPS)
- Model loading progress
- Generation progress bar
- Output paths

## Batch Size Recommendations

| Mac Model | Unified Memory | Recommended batch_size |
|-----------|----------------|------------------------|
| M1 (8 GB) | 8 GB | 16-24 |
| M2 (16 GB) | 16 GB | 24-32 |
| M3 (24+ GB) | 24+ GB | 32-48 |

## If MPS Fails

Code automatically falls back to CPU:
```
Warning: MPS placement failed (...), falling back to CPU
Using CPU device
```

This is safe but slower. Try reducing `--batch_size`.

## Next Steps

Once test succeeds, generate full dataset:
```bash
python generate_dataset.py \
  --model imagenet \
  --checkpoint_path ../checkpoints/imagenet_*.pth \
  --num_samples 5000 \
  --batch_size 32 \
  --samples_per_class 5 \
  --device mps
```
