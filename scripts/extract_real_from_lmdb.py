"""
Extract real images from LMDB dataset for FID evaluation.
Uses the correct LMDB key format: {array_name}_{idx}_data
"""

import argparse
import lmdb
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def get_array_shape_from_lmdb(lmdb_env, array_name):
    """Get array shape from LMDB metadata."""
    with lmdb_env.begin() as txn:
        shape_str = txn.get(f"{array_name}_shape".encode()).decode()
        shape = tuple(map(int, shape_str.split()))
    return shape


def retrieve_row_from_lmdb(lmdb_env, array_name, dtype, shape, row_index):
    """Retrieve a specific row from LMDB using correct key format."""
    data_key = f'{array_name}_{row_index}_data'.encode()

    with lmdb_env.begin() as txn:
        row_bytes = txn.get(data_key)

    if row_bytes is None:
        return None

    array = np.frombuffer(row_bytes, dtype=dtype)

    if len(shape) > 0:
        array = array.reshape(shape)
    return array


def extract_images(lmdb_path: str, output_dir: str, num_samples: int, seed: int = 42):
    """Extract images from LMDB to PNG files for FID evaluation."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Open LMDB
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    # Get shapes
    image_shape = get_array_shape_from_lmdb(env, 'images')
    label_shape = get_array_shape_from_lmdb(env, 'labels')

    total_images = image_shape[0]
    per_image_shape = image_shape[1:]  # (3, 64, 64) or (64, 64, 3)

    print(f"LMDB contains {total_images:,} images")
    print(f"Per-image shape: {per_image_shape}")
    print(f"Extracting {num_samples:,} images to {output_path}")

    # Random sample indices
    np.random.seed(seed)
    if num_samples >= total_images:
        indices = np.arange(total_images)
    else:
        indices = np.random.choice(total_images, size=num_samples, replace=False)

    extracted = 0
    for i, idx in enumerate(tqdm(indices, desc="Extracting")):
        # Get image using correct key format
        image = retrieve_row_from_lmdb(
            env, "images", np.uint8, per_image_shape, idx
        )

        if image is None:
            print(f"Warning: Could not retrieve image at index {idx}")
            continue

        # Handle different channel orderings
        if image.shape[0] == 3:  # CHW format
            image = image.transpose(1, 2, 0)  # -> HWC

        # Save as PNG
        img_path = output_path / f"real_{extracted:06d}.png"
        Image.fromarray(image).save(img_path)
        extracted += 1

    env.close()
    print(f"\nExtracted {extracted} images to {output_path}")
    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract real images from LMDB for FID evaluation")
    parser.add_argument("--lmdb_path", type=str, required=True, help="Path to LMDB directory")
    parser.add_argument("--output_dir", type=str, default="real_images", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of images to extract")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    extract_images(
        lmdb_path=args.lmdb_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
