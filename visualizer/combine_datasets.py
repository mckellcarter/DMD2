"""
Combine multiple extracted activation datasets into one.

Usage:
    python combine_datasets.py --datasets data/run1 data/run2 --output data/combined
    python combine_datasets.py --datasets data/run1 data/run2 --output data/combined --skip-images
    python combine_datasets.py --datasets data/run1 data/run2 --output data/combined -n 1000
"""

import argparse
from pathlib import Path

from data_sources import combine_datasets


def main():
    """CLI entry point for combining datasets."""
    parser = argparse.ArgumentParser(
        description="Combine multiple extracted activation datasets into one"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Paths to dataset directories to combine"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for combined dataset"
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip copying images (only combine activations and metadata)"
    )
    parser.add_argument(
        "-n", "--n-samples", "--max_samples_per_dataset",
        type=int,
        default=None,
        help="Take only first N samples from each dataset"
    )

    args = parser.parse_args()

    dataset_paths = [Path(p) for p in args.datasets]
    output_dir = Path(args.output)

    # Validate inputs
    for path in dataset_paths:
        metadata = path / "metadata" / "imagenet_real" / "dataset_info.json"
        if not metadata.exists():
            parser.error(f"Dataset metadata not found: {metadata}")

    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory not empty: {output_dir}")

    # Combine
    combine_datasets(
        dataset_paths=dataset_paths,
        output_dir=output_dir,
        copy_images=not args.skip_images,
        max_samples_per_dataset=args.n_samples
    )


if __name__ == "__main__":
    main()
