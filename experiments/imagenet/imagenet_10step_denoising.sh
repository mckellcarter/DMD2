#!/bin/bash
# ImageNet 64x64 10-Step Denoising Training
#
# This script trains a 10-step denoising model using backward simulation
# with optional classifier-free guidance support.
#
# Usage:
#   ./imagenet_10step_denoising.sh <CHECKPOINT_PATH> <WANDB_ENTITY> <WANDB_PROJECT>
#
# Arguments:
#   CHECKPOINT_PATH: Base path containing model checkpoints and datasets
#   WANDB_ENTITY: Your W&B username or team name
#   WANDB_PROJECT: W&B project name
#
# Required files in CHECKPOINT_PATH:
#   - edm-imagenet-64x64-cond-adm.pkl (pretrained EDM teacher model)
#   - imagenet-64x64_lmdb/ (ImageNet-64 dataset in LMDB format)
#
# Optional (for fine-tuning from 1-step):
#   - 1step_checkpoint/ (directory containing pytorch_model.bin)

export CHECKPOINT_PATH=$1
export WANDB_ENTITY=$2
export WANDB_PROJECT=$3

if [ -z "$CHECKPOINT_PATH" ] || [ -z "$WANDB_ENTITY" ] || [ -z "$WANDB_PROJECT" ]; then
    echo "Usage: $0 <CHECKPOINT_PATH> <WANDB_ENTITY> <WANDB_PROJECT>"
    exit 1
fi

# 8 GPU training for 10-step denoising
# Adjust --nproc_per_node based on available GPUs
torchrun --nproc_per_node 8 --nnodes 1 main/edm/train_edm_multistep.py \
    --generator_lr 5e-7 \
    --guidance_lr 5e-7 \
    --train_iters 500000 \
    --output_path $CHECKPOINT_PATH/imagenet_10step_denoising \
    --batch_size 32 \
    --initialie_generator \
    --log_iters 500 \
    --resolution 64 \
    --label_dim 1000 \
    --dataset_name "imagenet" \
    --seed 10 \
    --model_id $CHECKPOINT_PATH/edm-imagenet-64x64-cond-adm.pkl \
    --wandb_iters 100 \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --use_fp16 \
    --wandb_name "imagenet_10step_denoising" \
    --real_image_path $CHECKPOINT_PATH/imagenet-64x64_lmdb \
    --dfake_gen_update_ratio 5 \
    --cls_loss_weight 1e-2 \
    --gan_classifier \
    --gen_cls_loss_weight 3e-3 \
    --diffusion_gan \
    --diffusion_gan_max_timestep 1000 \
    --delete_ckpts \
    --max_checkpoint 10 \
    --denoising \
    --num_denoising_step 10 \
    --backward_simulation \
    --label_dropout 0.1

# Notes:
# - Remove --backward_simulation to use real images instead of generator outputs
# - Adjust --generator_lr to 2e-6 for training from scratch (vs 5e-7 for fine-tuning)
# - Add --ckpt_only_path for fine-tuning from existing checkpoint
# - Reduce --batch_size if running out of GPU memory
