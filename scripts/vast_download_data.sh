#!/bin/bash
# Download required data for DMD2 training
# Run this on vast.ai instance before training
#
# Usage: ./scripts/vast_download_data.sh [data_path] [hf_dataset_repo]
#
# Example:
#   ./scripts/vast_download_data.sh /data mckell/imagenet-64-lmdb

DATA_PATH=${1:-"/data"}
HF_DATASET_REPO=${2:-""}  # e.g., "mckell/imagenet-64-lmdb"

echo "======================================"
echo "Downloading DMD2 Training Data"
echo "======================================"
echo "Data Path: $DATA_PATH"
echo "HF Dataset Repo: ${HF_DATASET_REPO:-'(not specified)'}"
echo "======================================"


mkdir -p $DATA_PATH
cd $DATA_PATH

# Download pretrained EDM teacher model
if [ ! -f "edm-imagenet-64x64-cond-adm.pkl" ]; then
    echo "Downloading EDM teacher model..."
    wget -c https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl
else
    echo "EDM teacher model already exists"
fi

# Download FID reference statistics
if [ ! -f "VIRTUAL_imagenet64_labeled.npz" ]; then
    echo "Downloading FID reference stats..."
    wget -c https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/VIRTUAL_imagenet64_labeled.npz
else
    echo "FID reference stats already exist"
fi

# Download Inception network for FID
if [ ! -f "inception-2015-12-05.pkl" ]; then
    echo "Downloading Inception network..."
    wget -c https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pkl
else
    echo "Inception network already exists"
fi

# Download ImageNet-64 LMDB from HuggingFace (if repo specified)
if [ -n "$HF_DATASET_REPO" ]; then
    if [ ! -d "imagenet-64x64_lmdb" ] || [ ! -f "imagenet-64x64_lmdb/data.mdb" ]; then
        echo "Downloading ImageNet-64 LMDB from HuggingFace: $HF_DATASET_REPO"

        # Install huggingface_hub if needed
        pip install -q huggingface_hub

        # Download dataset
        huggingface-cli download "$HF_DATASET_REPO" \
            --repo-type dataset \
            --local-dir imagenet-64x64_lmdb \
            --local-dir-use-symlinks False

        echo "ImageNet-64 LMDB downloaded successfully"
    else
        echo "ImageNet-64 LMDB already exists"
    fi
else
    echo ""
    echo "NOTE: HuggingFace dataset repo not specified."
    echo "To auto-download, pass repo as second argument:"
    echo "  ./scripts/vast_download_data.sh /data your-username/imagenet-64-lmdb"
fi

echo "======================================"
echo "Data download complete!"
echo "======================================"

# Verify all required files
echo ""
echo "Checking required files:"
[ -f "edm-imagenet-64x64-cond-adm.pkl" ] && echo "  ✓ Teacher model" || echo "  ✗ Teacher model MISSING"
[ -f "VIRTUAL_imagenet64_labeled.npz" ] && echo "  ✓ FID reference stats" || echo "  ✗ FID reference stats MISSING"
[ -f "inception-2015-12-05.pkl" ] && echo "  ✓ Inception network" || echo "  ✗ Inception network MISSING"
[ -d "imagenet-64x64_lmdb" ] && [ -f "imagenet-64x64_lmdb/data.mdb" ] && echo "  ✓ ImageNet-64 LMDB" || echo "  ✗ ImageNet-64 LMDB MISSING"

echo ""
ls -la $DATA_PATH/
