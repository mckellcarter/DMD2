#!/bin/bash
# Vast.ai Activation Extraction Script
#
# Extracts activations from ImageNet-64 LMDB data using DMD2 model.
# Outputs can be used for UMAP visualization.
#
# Prerequisites:
#   - Model checkpoint (uploaded or downloaded)
#   - LMDB dataset (uploaded, e.g., imagenet-64x64_lmdb/)
#
# Usage:
#   ./scripts/vast_extract.sh <checkpoint_path> <lmdb_path> [output_dir] [num_samples] [batch_size]
#
# Examples:
#   ./scripts/vast_extract.sh /data/model /data/imagenet-64x64_lmdb
#   ./scripts/vast_extract.sh /data/checkpoint_029000 /data/imagenet-64x64_lmdb /outputs 10000 64

set -e

CHECKPOINT_PATH=${1:-"/data/model"}
LMDB_PATH=${2:-"/data/imagenet-64x64_lmdb"}
OUTPUT_DIR=${3:-"/outputs/activations"}
NUM_SAMPLES=${4:-10000}
BATCH_SIZE=${5:-64}

echo "======================================"
echo "DMD2 Activation Extraction (LMDB)"
echo "======================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "LMDB Path: $LMDB_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Num Samples: $NUM_SAMPLES"
echo "Batch Size: $BATCH_SIZE"
echo "======================================"

# Check GPU
echo "Checking GPU..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Validate checkpoint
if [ ! -e "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ -d "$CHECKPOINT_PATH" ]; then
    if [ ! -f "$CHECKPOINT_PATH/model.safetensors" ] && [ ! -f "$CHECKPOINT_PATH/pytorch_model.bin" ]; then
        echo "ERROR: No model file in checkpoint directory"
        echo "Expected: model.safetensors or pytorch_model.bin"
        exit 1
    fi
    echo "Checkpoint: directory format ✓"
else
    echo "Checkpoint: file format ✓"
fi

# Validate LMDB
if [ ! -d "$LMDB_PATH" ]; then
    echo "ERROR: LMDB directory not found: $LMDB_PATH"
    exit 1
fi

if [ ! -f "$LMDB_PATH/data.mdb" ]; then
    echo "ERROR: No data.mdb found in $LMDB_PATH"
    echo "Expected LMDB format with data.mdb file"
    exit 1
fi
echo "LMDB dataset found ✓"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Navigate to code
cd /workspace/DMD2

echo "======================================"
echo "Starting Extraction"
echo "======================================"

python visualizer/extract_real_imagenet.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --lmdb_path "$LMDB_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --layers encoder_bottleneck,midblock \
    --device cuda

echo "======================================"
echo "Extraction Complete!"
echo "======================================"
echo "Activations: $OUTPUT_DIR/activations/imagenet_real/"
echo "Metadata: $OUTPUT_DIR/metadata/imagenet_real/dataset_info.json"
echo ""
echo "Next: Run UMAP with ./scripts/vast_umap.sh $OUTPUT_DIR"
