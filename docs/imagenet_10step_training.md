# ImageNet 10-Step Denoising Training

Train a 10-step denoising diffusion model for ImageNet-64 using the DMD2 framework with backward simulation and classifier-free guidance (CFG).

## Overview

This extends the original single-step DMD2 approach to multi-step denoising, trading off inference speed for improved image quality. The model uses:

- **10 denoising steps** with Karras sigma schedule
- **Backward simulation** to reduce train-inference mismatch
- **CFG support** via label dropout during training
- **Distribution matching loss** at each denoising step

## Architecture

```
Training:
  noise -> backward_sim -> clean_image -> add_noise(σ_i) -> generator -> pred_image -> DM loss

Inference:
  noise * σ_max -> 10 CFG-guided steps -> final_image
```

## Prerequisites

1. **Dataset**: ImageNet-64 in LMDB format
2. **Teacher model**: `edm-imagenet-64x64-cond-adm.pkl`
3. **Hardware**: 8 GPUs recommended (~70 hours training)

## vast.ai Deployment

### Instance Requirements

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPUs | 4-8x A100/H100 | Multi-GPU training via torchrun |
| Instance Disk | **200GB** | Data (~20GB) + checkpoints (~80GB for 10 × 8GB) + buffer |
| Docker Image | ~76GB | Pulled to host cache, not instance disk |

### Environment Variables

Set these in vast.ai instance config:
- `WANDB_API_KEY` (required): Get from https://wandb.ai/authorize
- `HF_TOKEN` (optional): Only if using private HuggingFace repos

### Launch Steps

1. Create instance with image `mckellcarter/dmd2-train:latest`
2. SSH into instance
3. Run:
   ```bash
   cd /workspace/DMD2
   ./scripts/vast_startup.sh
   ```

The startup script auto-downloads teacher model, FID stats, and dataset from HuggingFace.

## Quick Start

### Training

```bash
# Basic training (from scratch)
./experiments/imagenet/imagenet_10step_denoising.sh /path/to/checkpoints your_wandb_entity your_project

# Fine-tune from 1-step checkpoint (faster convergence)
torchrun --nproc_per_node 8 main/edm/train_edm_multistep.py \
    --generator_lr 5e-7 \
    --ckpt_only_path /path/to/1step_checkpoint/ \
    --denoising \
    --num_denoising_step 10 \
    --backward_simulation \
    --label_dropout 0.1 \
    ... # other args
```

### Sampling

```bash
# Generate samples with CFG
python main/edm/sample_edm_multistep.py \
    --checkpoint /path/to/checkpoint/pytorch_model.bin \
    --num_samples 64 \
    --guidance_scale 1.5 \
    --num_steps 10 \
    --save_grid
```

### Evaluation

```bash
# FID evaluation
python main/edm/test_folder_edm_multistep.py \
    --folder /path/to/experiment \
    --num_denoising_step 10 \
    --guidance_scale 1.5 \
    --ref_path /path/to/imagenet64_ref_stats.npz \
    --detector_url /path/to/inception_v3.pkl
```

## Key Parameters

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--denoising` | flag | Enable multi-step denoising mode |
| `--num_denoising_step` | 10 | Number of denoising steps |
| `--backward_simulation` | flag | Use backward sim for training data |
| `--label_dropout` | 0.1 | Dropout rate for CFG training |
| `--generator_lr` | 5e-7 | Generator learning rate (lower for fine-tune) |
| `--guidance_lr` | 5e-7 | Guidance model learning rate |
| `--batch_size` | 32 | Per-GPU batch size |
| `--dfake_gen_update_ratio` | 5 | Generator update frequency |

### Inference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--guidance_scale` | 1.5 | CFG scale (1.0 = no guidance) |
| `--num_steps` | 10 | Number of denoising steps |
| `--sigma_max` | 80.0 | Starting noise level |
| `--sigma_min` | 0.002 | Ending noise level |
| `--rho` | 7.0 | Karras schedule parameter |

## Karras Sigma Schedule

The 10-step schedule uses the Karras formula:

```python
σ_i = (σ_max^(1/ρ) + i/(n-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
```

With default parameters (σ_max=80, σ_min=0.002, ρ=7):
```
[80.0, 35.1, 15.4, 6.73, 2.95, 1.29, 0.57, 0.25, 0.11, 0.002]
```

## Backward Simulation

During training, instead of using real images, we generate synthetic "clean" images by running the generator from noise:

```python
# Sample a random stopping point
selected_step = randint(0, num_steps)

# Run generator iteratively
x = noise * σ_max
for step in range(selected_step):
    pred = generator(x, σ[step], labels)
    x = pred + σ[step+1] * randn_like(pred)

# Use pred as clean image for training
```

This reduces the mismatch between what the model sees during training (real images) and inference (its own predictions).

## Classifier-Free Guidance

CFG combines conditional and unconditional predictions:

```python
pred = pred_uncond + scale * (pred_cond - pred_uncond)
```

Training with `--label_dropout 0.1` teaches the model to handle both cases:
- 90% of time: see actual class label
- 10% of time: see zeros (unconditional)

At inference, use `--guidance_scale` > 1.0 for sharper, more class-specific images.

## File Structure

```
main/edm/
├── edm_unified_model_multistep.py  # Multi-step model with backward sim
├── train_edm_multistep.py          # Training script
├── sample_edm_multistep.py         # Sampling with CFG
├── test_folder_edm_multistep.py    # FID evaluation
└── edm_network.py                  # Modified for label_dropout

experiments/imagenet/
└── imagenet_10step_denoising.sh    # Training launch script

tests/
└── test_edm_multistep.py           # Unit tests
```

## Training Tips

1. **Memory**: 10-step backward sim uses more memory. Reduce batch size if needed.
2. **Fine-tuning**: Start from 1-step checkpoint with lower LR (5e-7 vs 2e-6).
3. **CFG**: Use label_dropout=0.1 for good CFG/non-CFG balance.
4. **Monitoring**: Watch `loss_dm` and `gen_cls_loss` in W&B.

## Expected Results

| Config | Steps | CFG | Expected FID |
|--------|-------|-----|--------------|
| 1-step baseline | 1 | 1.0 | ~2.0 |
| 10-step | 10 | 1.0 | ~1.5-1.8 |
| 10-step + CFG | 10 | 1.5 | ~1.2-1.5 |

*FID numbers are estimates and may vary based on training duration and hyperparameters.*

## Troubleshooting

**Out of memory**: Reduce `--batch_size` or disable `--backward_simulation`.

**Divergence**: Use BF16 instead of FP16 (`--use_fp16` uses BF16 internally).

**Slow training**: Reduce `--num_denoising_step` for faster iteration.

**Poor CFG quality**: Increase `--label_dropout` to 0.15-0.2.

## References

- DMD2 Paper: [Distribution Matching Distillation](https://arxiv.org/abs/2403.03807)
- EDM Paper: [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
- CFG Paper: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
