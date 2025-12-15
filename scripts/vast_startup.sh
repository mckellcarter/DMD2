#!/bin/bash
# Vast.ai Startup Script for DMD2 10-Step Training
#
# Set this as your "on-start script" in vast.ai, or run manually after connecting.
#
# Prerequisites:
#   - Set WANDB_API_KEY environment variable in vast.ai
#   - Set HF_TOKEN environment variable if using private HuggingFace repos
#
# Usage:
#   ./scripts/vast_startup.sh <wandb_entity> <wandb_project> [hf_dataset_repo] [num_gpus]
#
# Example:
#   ./scripts/vast_startup.sh myteam dmd2_10step mckell/imagenet-64-lmdb 8

set -e

WANDB_ENTITY=${1:-"mckellcarter-university-of-colorado-boulder-org"}
WANDB_PROJECT=${2:-"dmd2_10step"}
HF_DATASET_REPO=${3:-"mckell/imagenet-64-lmdb"}
NUM_GPUS=${4:-8}

DATA_PATH="/data"
OUTPUT_PATH="/outputs"

echo "======================================"
echo "DMD2 10-Step Training Setup"
echo "======================================"
echo "W&B Entity: $WANDB_ENTITY"
echo "W&B Project: $WANDB_PROJECT"
echo "HF Dataset: ${HF_DATASET_REPO:-'(manual upload)'}"
echo "Num GPUs: $NUM_GPUS"
echo "Data Path: $DATA_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "======================================"

# Download all required data (teacher model, FID stats, dataset if HF repo specified)
echo "Downloading required data..."
cd /workspace/DMD2
./scripts/vast_download_data.sh "$DATA_PATH" "$HF_DATASET_REPO"

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f "$DATA_PATH/edm-imagenet-64x64-cond-adm.pkl" ]; then
    echo "ERROR: Teacher model not found at $DATA_PATH/edm-imagenet-64x64-cond-adm.pkl"
    exit 1
fi

if [ ! -d "$DATA_PATH/imagenet-64x64_lmdb" ] || [ ! -f "$DATA_PATH/imagenet-64x64_lmdb/data.mdb" ]; then
    echo "ERROR: ImageNet-64 LMDB not found at $DATA_PATH/imagenet-64x64_lmdb/"
    echo "Either specify HF_DATASET_REPO or upload manually"
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. W&B logging may fail."
    echo "Set it in vast.ai environment variables or run: wandb login"
fi

# Check GPU availability
echo "Checking GPUs..."
nvidia-smi --query-gpu=name,memory.total --format=csv
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $GPU_COUNT GPUs"

if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
    echo "WARNING: Requested $NUM_GPUS GPUs but only $GPU_COUNT available"
    NUM_GPUS=$GPU_COUNT
fi

# Create output directories
mkdir -p $OUTPUT_PATH/imagenet_10step_denoising

# Navigate to code and set PYTHONPATH
cd /workspace/DMD2
export PYTHONPATH=/workspace/DMD2:$PYTHONPATH

# Run tests to verify installation
echo "Running quick tests..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
from main.edm.edm_unified_model_multistep import get_denoising_sigmas
sigmas = get_denoising_sigmas(10, 80.0, 0.002)
print(f'Sigma schedule OK: {len(sigmas)} steps')
"

echo "======================================"
echo "Starting Training"
echo "======================================"

# Launch training
# NOTE: batch_size=4 works for RTX 3090 (24GB). For A100 (80GB), try batch_size=16
# NOTE: Mixed precision (fp16/bf16) causes dtype issues with EDM network, using fp32
torchrun --nproc_per_node $NUM_GPUS --nnodes 1 main/edm/train_edm_multistep.py \
    --generator_lr 5e-7 \
    --guidance_lr 5e-7 \
    --train_iters 500000 \
    --output_path $OUTPUT_PATH/imagenet_10step_denoising \
    --batch_size 4 \
    --initialie_generator \
    --log_iters 500 \
    --resolution 64 \
    --label_dim 1000 \
    --dataset_name "imagenet" \
    --seed 10 \
    --model_id $DATA_PATH/edm-imagenet-64x64-cond-adm.pkl \
    --wandb_iters 100 \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --wandb_name "imagenet_10step_$(date +%Y%m%d_%H%M%S)" \
    --real_image_path $DATA_PATH/imagenet-64x64_lmdb \
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
