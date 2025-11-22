"""
Process activations into UMAP embeddings for visualization.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from umap import UMAP
from sklearn.preprocessing import StandardScaler

from extract_activations import load_activations, flatten_activations


def load_dataset_activations(
    activation_dir: Path,
    metadata_path: Path,
    max_samples: int = None
):
    """
    Load all activations from dataset.

    Args:
        activation_dir: Directory containing activation .npz files
        metadata_path: Path to dataset_info.json
        max_samples: Optional limit on samples to load

    Returns:
        (activation_matrix, metadata_df)
        - activation_matrix: (N, D) array of flattened activations
        - metadata_df: DataFrame with sample info
    """
    # Load metadata
    with open(metadata_path, 'r') as f:
        dataset_info = json.load(f)

    samples = dataset_info['samples'][:max_samples] if max_samples else dataset_info['samples']

    print(f"Loading {len(samples)} samples...")

    all_activations = []
    metadata_records = []

    for sample in tqdm(samples, desc="Loading activations"):
        sample_id = sample['sample_id']
        act_path = activation_dir / f"{sample_id}.npz"

        if not act_path.exists():
            print(f"Warning: Missing {act_path}")
            continue

        # Load activations
        activations, _ = load_activations(act_path)

        # Flatten to 1D vector
        flat_act = flatten_activations(activations)
        all_activations.append(flat_act[0])  # Remove batch dim

        # Store metadata
        metadata_records.append(sample)

    # Stack into matrix
    activation_matrix = np.stack(all_activations, axis=0)
    metadata_df = pd.DataFrame(metadata_records)

    print(f"Loaded activations: {activation_matrix.shape}")
    return activation_matrix, metadata_df


def compute_umap(
    activations: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    n_components: int = 2,
    random_state: int = 42,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute UMAP projection.

    Args:
        activations: (N, D) activation matrix
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric
        n_components: Output dimensions (2 or 3)
        random_state: Random seed
        normalize: Whether to normalize before UMAP

    Returns:
        (N, n_components) UMAP coordinates
    """
    print(f"\nComputing UMAP with:")
    print(f"  n_neighbors={n_neighbors}")
    print(f"  min_dist={min_dist}")
    print(f"  metric={metric}")
    print(f"  n_components={n_components}")

    # Normalize activations
    if normalize:
        print("Normalizing activations...")
        scaler = StandardScaler()
        activations = scaler.fit_transform(activations)

    # Compute UMAP
    print("Running UMAP...")
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=n_components,
        random_state=random_state,
        verbose=True
    )

    embeddings = reducer.fit_transform(activations)

    print(f"UMAP embeddings: {embeddings.shape}")
    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    output_path: Path,
    umap_params: dict
):
    """
    Save UMAP embeddings + metadata to CSV.

    Args:
        embeddings: (N, 2) or (N, 3) UMAP coordinates
        metadata_df: DataFrame with sample metadata
        output_path: Output CSV path
        umap_params: Dict of UMAP parameters
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create output dataframe
    df = metadata_df.copy()

    # Add UMAP coordinates
    if embeddings.shape[1] == 2:
        df['umap_x'] = embeddings[:, 0]
        df['umap_y'] = embeddings[:, 1]
    else:
        df['umap_x'] = embeddings[:, 0]
        df['umap_y'] = embeddings[:, 1]
        df['umap_z'] = embeddings[:, 2]

    # Save CSV
    df.to_csv(output_path, index=False)
    print(f"\nSaved embeddings to {output_path}")

    # Save UMAP parameters
    param_path = output_path.with_suffix('.json')
    with open(param_path, 'w') as f:
        json.dump(umap_params, f, indent=2)
    print(f"Saved parameters to {param_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process DMD2 activations into UMAP embeddings"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["imagenet", "sdxl", "sdv1.5"],
        help="Model type"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data directory containing activations"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/embeddings",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="all",
        help="Layer name (currently uses all layers)"
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter"
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="Distance metric for UMAP"
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of UMAP dimensions"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples to process"
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Skip normalization before UMAP"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for UMAP"
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    activation_dir = data_dir / "activations" / args.model
    metadata_path = data_dir / "metadata" / args.model / "dataset_info.json"
    output_dir = Path(args.output_dir)

    # Load activations
    activations, metadata_df = load_dataset_activations(
        activation_dir,
        metadata_path,
        max_samples=args.max_samples
    )

    # Compute UMAP
    embeddings = compute_umap(
        activations,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        n_components=args.n_components,
        random_state=args.random_state,
        normalize=not args.no_normalize
    )

    # Save results
    output_name = f"{args.model}_umap_n{args.n_neighbors}_d{args.min_dist}.csv"
    output_path = output_dir / output_name

    umap_params = {
        "model": args.model,
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "metric": args.metric,
        "n_components": args.n_components,
        "normalize": not args.no_normalize,
        "random_state": args.random_state,
        "num_samples": len(activations)
    }

    save_embeddings(embeddings, metadata_df, output_path, umap_params)

    print("\nDone!")


if __name__ == "__main__":
    main()
