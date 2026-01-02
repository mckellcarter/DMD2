"""
Extract activations from real ImageNet images using DMD2 model.
Supports LMDB, NPZ, and JPEG directory formats.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
from tqdm import tqdm
from accelerate.utils import set_seed

# Local imports
from extract_activations import ActivationExtractor
from device_utils import get_device, get_device_info
from model_utils import create_imagenet_generator
from data_sources import (
    create_data_source,
    BatchProcessor,
    load_class_labels_map,
    create_output_dirs
)


def batched(iterable, n):
    """Batch an iterable into chunks of size n."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def extract_real_imagenet_activations(
    checkpoint_path: str,
    imagenet_dir: Optional[Path],
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    layers: List[str],
    conditioning_sigma: float = 80.0,
    split: str = "train",
    seed: int = 10,
    device: str = None,
    npz_dir: Optional[Path] = None,
    lmdb_path: Optional[Path] = None,
    num_classes: int = 1000,
    target_classes: Optional[List[int]] = None
):
    """
    Extract activations from real ImageNet images.

    Args:
        checkpoint_path: Path to DMD2 model checkpoint
        imagenet_dir: Root directory of ImageNet dataset (for JPEG format)
        output_dir: Output directory for activations
        num_samples: Total samples to process
        batch_size: Batch size for processing
        layers: List of layer names to extract
        conditioning_sigma: Sigma for forward pass (default: 80.0)
        split: Dataset split ("val" or "train")
        seed: Random seed for shuffling
        device: Device to use (auto-detect if None)
        npz_dir: Directory containing ImageNet64 NPZ files
        lmdb_path: Path to LMDB dataset
        num_classes: Number of classes to sample from
        target_classes: Specific class IDs to sample from
    """
    # Auto-detect device
    if device is None:
        device = get_device()

    device_info = get_device_info(device)
    print(f"\nDevice: {device_info['device']} ({device_info['device_name']})")
    if device_info['memory_allocated'] != 'N/A':
        print(f"GPU Memory: {device_info['memory_allocated']:.2f} GB allocated")

    # Load model
    print(f"\nLoading ImageNet model from {checkpoint_path}")
    generator = create_imagenet_generator(checkpoint_path, device)

    # Setup extractor
    extractor = ActivationExtractor(model_type="imagenet")
    extractor.register_hooks(generator, layers)

    # Load class labels map
    class_labels_map = load_class_labels_map()

    # Create output directories
    output_dirs = create_output_dirs(output_dir)

    # Select target classes
    set_seed(seed)
    if target_classes is None:
        target_classes = sorted(
            np.random.choice(1000, size=min(num_classes, 1000), replace=False).tolist()
        )
    else:
        target_classes = sorted(target_classes)

    samples_per_class = num_samples // len(target_classes)
    print(f"\nSampling from {len(target_classes)} classes")
    print(f"Target: ~{samples_per_class} samples per class")

    # Determine source type for metadata
    if lmdb_path:
        source_name = "imagenet_real_lmdb"
    elif npz_dir:
        source_name = "imagenet_real_npz"
    else:
        source_name = "imagenet_real"

    # Create data source
    with create_data_source(
        lmdb_path=lmdb_path,
        npz_dir=npz_dir,
        imagenet_dir=imagenet_dir,
        split=split,
        class_labels_map=class_labels_map
    ) as source:
        # Scan for samples matching criteria
        selected_indices = source.scan_samples(target_classes, samples_per_class, num_samples)

        # Create batch processor
        processor = BatchProcessor(
            extractor=extractor,
            generator=generator,
            output_dirs=output_dirs,
            class_labels_map=class_labels_map,
            device=device,
            conditioning_sigma=conditioning_sigma,
            layers=layers,
            source_name=source_name
        )

        # Process batches
        all_metadata = []
        sample_idx = 0
        num_batches = (len(selected_indices) + batch_size - 1) // batch_size

        for batch_idx, batch_indices in enumerate(
            tqdm(batched(selected_indices, batch_size), total=num_batches, desc="Processing batches")
        ):
            # Load batch
            images, labels, paths = source.load_batch(batch_indices)

            # Convert to standard labels
            labels_std = source.get_standard_labels(labels)

            # Process batch
            sample_idx, batch_meta = processor.process_batch(
                images, labels_std, paths, batch_idx, sample_idx
            )
            all_metadata.extend(batch_meta)

    # Save global metadata
    metadata_path = output_dirs['metadata'] / "dataset_info.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model_type": "imagenet_real",
            "num_samples": sample_idx,
            "layers": layers,
            "conditioning_sigma": conditioning_sigma,
            "seed": seed,
            "split": split,
            "class_labels": class_labels_map,
            "samples": all_metadata
        }, f, indent=2)

    print(f"\nProcessed {sample_idx} real ImageNet samples")
    print(f"Original Images: {output_dirs['image']}")
    print(f"Reconstructed Images: {output_dirs['reconstructed']}")
    print(f"Activations: {output_dirs['activation']}")
    print(f"Metadata: {metadata_path}")
    print(f"Conditioning Sigma: {conditioning_sigma}")

    extractor.remove_hooks()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Extract activations from real ImageNet images"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to DMD2 model checkpoint"
    )
    parser.add_argument(
        "--imagenet_dir",
        type=str,
        default=None,
        help="Root directory of ImageNet dataset (JPEG format)"
    )
    parser.add_argument(
        "--npz_dir",
        type=str,
        default=None,
        help="Directory containing ImageNet64 NPZ batch files"
    )
    parser.add_argument(
        "--lmdb_path",
        type=str,
        default=None,
        help="Path to ImageNet LMDB dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for activations"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to process"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="encoder_bottleneck,midblock",
        help="Comma-separated list of layers to extract"
    )
    parser.add_argument(
        "--conditioning_sigma",
        type=float,
        default=80.0,
        help="Conditioning sigma for forward pass (default: 80.0)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["val", "train"],
        help="ImageNet split to process"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (auto-detect if not specified)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Number of classes to sample from"
    )
    parser.add_argument(
        "--target_classes",
        type=str,
        default=None,
        help="Comma-separated list of class IDs to sample from"
    )

    args = parser.parse_args()

    # Validate input arguments
    sources = [args.imagenet_dir, args.npz_dir, args.lmdb_path]
    num_sources = sum(1 for s in sources if s is not None)

    if num_sources == 0:
        parser.error("One of --imagenet_dir, --npz_dir, or --lmdb_path must be provided")

    if num_sources > 1:
        parser.error("Cannot use multiple input sources. Choose one.")

    output_dir = Path(args.output_dir)
    imagenet_dir = Path(args.imagenet_dir) if args.imagenet_dir else None
    npz_dir = Path(args.npz_dir) if args.npz_dir else None
    lmdb_path = Path(args.lmdb_path) if args.lmdb_path else None
    layers = args.layers.split(",")

    # Parse target_classes
    target_classes = None
    if args.target_classes:
        target_classes = [int(c.strip()) for c in args.target_classes.split(",")]

    extract_real_imagenet_activations(
        checkpoint_path=args.checkpoint_path,
        imagenet_dir=imagenet_dir,
        output_dir=output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        layers=layers,
        conditioning_sigma=args.conditioning_sigma,
        split=args.split,
        seed=args.seed,
        device=args.device,
        npz_dir=npz_dir,
        lmdb_path=lmdb_path,
        num_classes=args.num_classes,
        target_classes=target_classes
    )


if __name__ == "__main__":
    main()
