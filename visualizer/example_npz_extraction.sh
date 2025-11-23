#!/bin/bash
# Example: Extract activations from ImageNet64 NPZ files
#
# This example demonstrates using the NPZ format for fast activation extraction.
# The NPZ format is 10-100x faster than loading individual JPEG files.

# Navigate to visualizer directory
cd "$(dirname "$0")"

# Configuration
CHECKPOINT_PATH="../checkpoints/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch_fid1.51_checkpoint_model_193500.pth"
NPZ_DIR="data/Imagenet64_train_npz"
OUTPUT_DIR="data"
NUM_SAMPLES=1000
BATCH_SIZE=128
LAYERS="encoder_bottleneck,midblock"
SEED=42

echo "========================================"
echo "ImageNet64 NPZ Activation Extraction"
echo "========================================"
echo ""
echo "Configuration:"
echo "  NPZ Directory: $NPZ_DIR"
echo "  Samples: $NUM_SAMPLES"
echo "  Batch Size: $BATCH_SIZE"
echo "  Layers: $LAYERS"
echo "  Seed: $SEED"
echo ""

# Check if NPZ directory exists
if [ ! -d "$NPZ_DIR" ]; then
    echo "ERROR: NPZ directory not found: $NPZ_DIR"
    echo "Please ensure ImageNet64 NPZ files are in the correct location."
    exit 1
fi

# Count NPZ files
NPZ_COUNT=$(ls -1 "$NPZ_DIR"/*.npz 2>/dev/null | wc -l)
echo "Found $NPZ_COUNT NPZ batch files"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "WARNING: Checkpoint not found: $CHECKPOINT_PATH"
    echo "Please download the checkpoint using:"
    echo "  cd .. && bash scripts/download_hf_checkpoint.sh \\"
    echo "    'imagenet/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch_fid1.51_checkpoint_model_193500' \\"
    echo "    checkpoints/"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Starting extraction..."
echo ""

# Run extraction
python extract_real_imagenet.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --npz_dir "$NPZ_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_samples $NUM_SAMPLES \
  --batch_size $BATCH_SIZE \
  --layers "$LAYERS" \
  --conditioning_sigma 0.0 \
  --seed $SEED \
  --device cuda

echo ""
echo "========================================"
echo "Extraction Complete!"
echo "========================================"
echo ""
echo "Output files:"
echo "  Original images: $OUTPUT_DIR/images/imagenet_real/"
echo "  Reconstructed:   $OUTPUT_DIR/images/imagenet_real_reconstructed/"
echo "  Activations:     $OUTPUT_DIR/activations/imagenet_real/"
echo "  Metadata:        $OUTPUT_DIR/metadata/imagenet_real/"
echo ""
echo "Next steps:"
echo "  1. Process embeddings: python process_embeddings.py --model imagenet_real"
echo "  2. Launch visualizer: ./run_visualizer.sh imagenet_real"
echo ""
