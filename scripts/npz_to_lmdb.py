"""
Convert ImageNet-64 NPZ files to LMDB format for DMD2 training.

Usage:
    python scripts/npz_to_lmdb.py \
        --npz_dir visualizer/data/Imagenet64_train_npz \
        --lmdb_path data/imagenet-64x64_lmdb

NPZ format (from image-net.org):
    - data: (N, 12288) uint8 - flattened RGB images
    - labels: (N,) int64 - class labels 1-1000 (ImageNet64 ordering)
    - mean: (12288,) float64 - mean image

LMDB format (for DMD2):
    - images: (N, 3, 64, 64) uint8 - CHW format
    - labels: (N, 1) int64 - class labels 0-999 (STANDARD ImageNet ordering)

IMPORTANT: ImageNet64 NPZ uses a different class ordering than standard ImageNet.
This script remaps labels to match standard ImageNet ordering used by pretrained models.
"""

import numpy as np
import argparse
import lmdb
import glob
import os
import sys
from tqdm import tqdm

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualizer.class_mapping import remap_imagenet64_labels_to_standard


def store_arrays_to_lmdb(env, arrays_dict, start_index=0):
    """Store rows of numpy arrays to LMDB."""
    with env.begin(write=True) as txn:
        for array_name, array in arrays_dict.items():
            for i, row in enumerate(array):
                row_bytes = row.tobytes()
                data_key = f'{array_name}_{start_index + i}_data'.encode()
                txn.put(data_key, row_bytes)


def convert_npz_to_lmdb(npz_dir, lmdb_path, map_size_gb=50):
    """Convert NPZ files to LMDB format."""

    # Find all NPZ files
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "train_data_batch_*.npz")))

    if len(npz_files) == 0:
        raise ValueError(f"No NPZ files found in {npz_dir}")

    print(f"Found {len(npz_files)} NPZ files")

    # Create LMDB environment
    map_size = map_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    os.makedirs(os.path.dirname(lmdb_path) if os.path.dirname(lmdb_path) else '.', exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=map_size)

    total_images = 0

    for npz_file in tqdm(npz_files, desc="Converting NPZ files"):
        # Load NPZ
        data = np.load(npz_file)

        # Get images and reshape from (N, 12288) to (N, 3, 64, 64)
        images = data['data'].reshape(-1, 3, 64, 64).astype(np.uint8)

        # Get labels: convert from 1-indexed to 0-indexed, then remap to standard ImageNet
        labels_0indexed = (data['labels'] - 1).astype(np.int64)
        labels_remapped = remap_imagenet64_labels_to_standard(labels_0indexed)
        labels = labels_remapped.reshape(-1, 1)

        # Verify shapes
        assert images.shape[1:] == (3, 64, 64), f"Unexpected image shape: {images.shape}"
        assert labels.min() >= 0 and labels.max() <= 999, f"Label range error: {labels.min()}-{labels.max()}"

        # Store to LMDB
        arrays_dict = {
            'images': images,
            'labels': labels
        }
        store_arrays_to_lmdb(env, arrays_dict, start_index=total_images)

        total_images += len(images)

        # Free memory
        del data, images, labels

    # Store shape metadata
    print(f"\nTotal images: {total_images:,}")
    print("Storing metadata...")

    with env.begin(write=True) as txn:
        # Images shape: (total, 3, 64, 64)
        images_shape = f"{total_images} 3 64 64"
        txn.put(b"images_shape", images_shape.encode())

        # Labels shape: (total, 1)
        labels_shape = f"{total_images} 1"
        txn.put(b"labels_shape", labels_shape.encode())

    env.close()
    print(f"LMDB created at: {lmdb_path}")

    # Verify
    print("\nVerifying LMDB...")
    verify_lmdb(lmdb_path)


def verify_lmdb(lmdb_path):
    """Verify the LMDB was created correctly."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    with env.begin() as txn:
        # Check shapes
        images_shape = txn.get(b"images_shape").decode()
        labels_shape = txn.get(b"labels_shape").decode()
        print(f"Images shape: {images_shape}")
        print(f"Labels shape: {labels_shape}")

        # Load first image
        img_bytes = txn.get(b"images_0_data")
        img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(3, 64, 64)
        print(f"First image shape: {img.shape}, range: [{img.min()}, {img.max()}]")

        # Load first label
        label_bytes = txn.get(b"labels_0_data")
        label = np.frombuffer(label_bytes, dtype=np.int64)
        print(f"First label: {label}")

        # Check a few random indices
        total = int(images_shape.split()[0])
        for idx in [0, 1000, total - 1]:
            img_key = f"images_{idx}_data".encode()
            label_key = f"labels_{idx}_data".encode()
            assert txn.get(img_key) is not None, f"Missing image at index {idx}"
            assert txn.get(label_key) is not None, f"Missing label at index {idx}"

    env.close()
    print("Verification passed!")


def main():
    parser = argparse.ArgumentParser(description="Convert ImageNet-64 NPZ to LMDB")
    parser.add_argument("--npz_dir", type=str, required=True,
                        help="Directory containing train_data_batch_*.npz files")
    parser.add_argument("--lmdb_path", type=str, required=True,
                        help="Output LMDB path")
    parser.add_argument("--map_size_gb", type=int, default=50,
                        help="LMDB map size in GB (default: 50)")

    args = parser.parse_args()

    convert_npz_to_lmdb(args.npz_dir, args.lmdb_path, args.map_size_gb)


if __name__ == "__main__":
    main()
