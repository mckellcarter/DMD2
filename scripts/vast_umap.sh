#!/bin/bash
# Vast.ai UMAP Processing Script
#
# Computes UMAP embeddings from DMD2 activations using GPU acceleration.
# Supports multi-GPU via Dask-cuML.
#
# Prerequisites:
#   - NVIDIA GPU(s) with RAPIDS cuML installed
#   - Activations already extracted to data directory
#
# Usage:
#   ./scripts/vast_umap.sh <data_path> [output_path] [n_neighbors] [min_dist] [num_gpus]
#
# Examples:
#   # Single GPU, default params
#   ./scripts/vast_umap.sh /data
#
#   # Multi-GPU with custom params
#   ./scripts/vast_umap.sh /data /outputs 25 0.1 4
#
#   # All GPUs
#   ./scripts/vast_umap.sh /data /outputs 25 0.1 all

set -e

DATA_PATH=${1:-"/data"}
OUTPUT_PATH=${2:-"$DATA_PATH/embeddings"}
N_NEIGHBORS=${3:-25}
MIN_DIST=${4:-0.1}
NUM_GPUS=${5:-"1"}

echo "======================================"
echo "DMD2 UMAP Processing (CUDA)"
echo "======================================"
echo "Data Path: $DATA_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "n_neighbors: $N_NEIGHBORS"
echo "min_dist: $MIN_DIST"
echo "Requested GPUs: $NUM_GPUS"
echo "======================================"

# Check for RAPIDS/cuML
echo "Checking RAPIDS installation..."
python -c "
import sys
try:
    import cuml
    print(f'cuML version: {cuml.__version__}')
except ImportError:
    print('ERROR: cuML not found. Install with: pip install cuml-cu12')
    sys.exit(1)

try:
    import cupy as cp
    gpu_count = cp.cuda.runtime.getDeviceCount()
    print(f'Available GPUs: {gpu_count}')
    for i in range(gpu_count):
        props = cp.cuda.runtime.getDeviceProperties(i)
        mem = cp.cuda.Device(i).mem_info[1] / 1e9
        print(f'  GPU {i}: {props[\"name\"].decode()} ({mem:.1f} GB)')
except Exception as e:
    print(f'GPU check failed: {e}')
    sys.exit(1)
"

# Check for activation data
if [ ! -d "$DATA_PATH/activations/imagenet_real" ]; then
    echo "ERROR: Activations not found at $DATA_PATH/activations/imagenet_real"
    echo "Run extract_real_imagenet.py first to generate activations."
    exit 1
fi

if [ ! -f "$DATA_PATH/metadata/imagenet_real/dataset_info.json" ]; then
    echo "ERROR: Metadata not found at $DATA_PATH/metadata/imagenet_real/dataset_info.json"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Navigate to code
cd /workspace/DMD2

# Determine GPU flags
GPU_FLAGS=""
if [ "$NUM_GPUS" = "all" ]; then
    GPU_FLAGS="--multi_gpu"
elif [ "$NUM_GPUS" -gt 1 ] 2>/dev/null; then
    GPU_FLAGS="--num_gpus $NUM_GPUS"
fi

echo "======================================"
echo "Starting UMAP Processing"
echo "======================================"

# Run UMAP
python visualizer/process_embeddings_cuda.py \
    --model imagenet_real \
    --data_dir "$DATA_PATH" \
    --output_dir "$OUTPUT_PATH" \
    --n_neighbors "$N_NEIGHBORS" \
    --min_dist "$MIN_DIST" \
    $GPU_FLAGS

echo "======================================"
echo "UMAP Complete!"
echo "======================================"
echo "Output: $OUTPUT_PATH/imagenet_real_umap_n${N_NEIGHBORS}_d${MIN_DIST}.csv"
