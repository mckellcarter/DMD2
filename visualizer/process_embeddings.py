"""
Process activations into UMAP embeddings for visualization.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import pickle
from umap import UMAP
from sklearn.preprocessing import StandardScaler

from extract_activations import load_activations, flatten_activations


def load_dataset_activations(
    activation_dir: Path,
    metadata_path: Path,
    max_samples: int = None,
    batch_size: int = 500,
    low_memory: bool = False
):
    """
    Load all activations from dataset using batched loading to reduce memory.

    Args:
        activation_dir: Directory containing activation .npz files
        metadata_path: Path to dataset_info.json
        max_samples: Optional limit on samples to load
        batch_size: Number of samples to load per batch
        low_memory: Use memory-mapped temp file (slower but handles large datasets)

    Returns:
        (activation_matrix, metadata_df)
        - activation_matrix: (N, D) array of flattened activations
        - metadata_df: DataFrame with sample info
    """
    import tempfile

    # Load metadata
    with open(metadata_path, 'r') as f:
        dataset_info = json.load(f)

    samples = dataset_info['samples'][:max_samples] if max_samples else dataset_info['samples']

    print(f"Loading {len(samples)} samples in batches of {batch_size}...")
    if low_memory:
        print("Using low-memory mode (memory-mapped file)")

    # First pass: determine activation shape
    first_sample = samples[0]

    # Handle both batch format (activation_path) and old format (sample_id)
    if 'activation_path' in first_sample:
        # New format: batch files
        # activation_path is relative to data root (e.g., "activations/imagenet_real/batch_000000")
        # activation_dir is "data/activations/imagenet_real", so data root is activation_dir.parent.parent
        data_root = activation_dir.parent.parent
        first_path = data_root / first_sample['activation_path']
        first_batch_index = first_sample['batch_index']
    else:
        # Old format: individual sample files
        first_path = activation_dir / f"{first_sample['sample_id']}.npz"
        first_batch_index = 0

    first_act, _ = load_activations(first_path)
    first_flat = flatten_activations(first_act)
    activation_dim = first_flat.shape[1]

    # For batch format, we only need the dimension from one sample
    # (first_batch_index is used to know which sample, but shape is the same)

    # Preallocate array (memory-mapped if low_memory)
    if low_memory:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        activation_matrix = np.memmap(
            temp_file.name,
            dtype=np.float32,
            mode='w+',
            shape=(len(samples), activation_dim)
        )
    else:
        activation_matrix = np.zeros((len(samples), activation_dim), dtype=np.float32)

    metadata_records = []

    # Load in batches
    valid_idx = 0
    batch_cache = {}  # Cache batch NPZ files to avoid redundant loads

    for i in tqdm(range(0, len(samples), batch_size), desc="Loading batches"):
        batch_samples = samples[i:i+batch_size]

        for sample in batch_samples:
            sample_id = sample['sample_id']

            # Get batch path and index from metadata
            if 'batch_index' in sample:
                # New format: batch files
                # activation_path is relative to data root
                act_path_str = sample['activation_path']
                batch_index = sample['batch_index']
                data_root = activation_dir.parent.parent
                act_path = data_root / act_path_str
            else:
                # Old format: individual sample files (backwards compatibility)
                act_path = activation_dir / f"{sample_id}.npz"
                batch_index = 0

            # Check if file exists (add .npz extension for batch format)
            if not act_path.with_suffix('.npz').exists():
                print(f"Warning: Missing {act_path.with_suffix('.npz')}")
                continue

            # Load activations (use cache for batch files)
            act_path_key = str(act_path)
            if act_path_key not in batch_cache:
                activations, _ = load_activations(act_path)
                batch_cache[act_path_key] = activations
            else:
                activations = batch_cache[act_path_key]

            # Flatten to 1D vector
            flat_act = flatten_activations(activations)
            activation_matrix[valid_idx] = flat_act[batch_index]

            # Store metadata
            metadata_records.append(sample)
            valid_idx += 1

    # Trim to valid samples
    if low_memory:
        # Copy to regular array and cleanup
        result = np.array(activation_matrix[:valid_idx])
        del activation_matrix
        import os
        os.unlink(temp_file.name)
        activation_matrix = result
    else:
        activation_matrix = activation_matrix[:valid_idx]

    metadata_df = pd.DataFrame(metadata_records)

    print(f"Loaded activations: {activation_matrix.shape}")

    # Check for NaN/inf values and handle them
    nan_count = np.isnan(activation_matrix).sum()
    inf_count = np.isinf(activation_matrix).sum()

    if nan_count > 0 or inf_count > 0:
        print(f"Warning: Found {nan_count} NaN and {inf_count} inf values")
        print("Replacing NaN/inf with 0...")
        activation_matrix = np.nan_to_num(activation_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return activation_matrix, metadata_df


def compute_umap(
    activations: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    n_components: int = 2,
    random_state: int = 42,
    normalize: bool = True
):
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
        (embeddings, reducer, scaler)
        - embeddings: (N, n_components) UMAP coordinates
        - reducer: Fitted UMAP model
        - scaler: Fitted StandardScaler (or None if not normalized)
    """
    print(f"\nComputing UMAP with:")
    print(f"  n_neighbors={n_neighbors}")
    print(f"  min_dist={min_dist}")
    print(f"  metric={metric}")
    print(f"  n_components={n_components}")

    # Normalize activations
    scaler = None
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
    return embeddings, reducer, scaler


def save_embeddings(
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    output_path: Path,
    umap_params: dict,
    reducer=None,
    scaler=None
):
    """
    Save UMAP embeddings + metadata to CSV.

    Args:
        embeddings: (N, 2) or (N, 3) UMAP coordinates
        metadata_df: DataFrame with sample metadata
        output_path: Output CSV path
        umap_params: Dict of UMAP parameters
        reducer: Fitted UMAP model (for inverse_transform)
        scaler: Fitted StandardScaler (for inverse_transform)
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

    # Save UMAP model and scaler for inverse_transform
    if reducer is not None:
        model_path = output_path.with_suffix('.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'reducer': reducer,
                'scaler': scaler
            }, f)
        print(f"Saved UMAP model to {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process DMD2 activations into UMAP embeddings"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["imagenet", "imagenet_real", "sdxl", "sdv1.5"],
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
    parser.add_argument(
        "--low_memory",
        action="store_true",
        help="Use memory-mapped file for large datasets (slower but uses less RAM)"
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
        max_samples=args.max_samples,
        low_memory=args.low_memory
    )

    # Compute UMAP
    embeddings, reducer, scaler = compute_umap(
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

    save_embeddings(embeddings, metadata_df, output_path, umap_params, reducer, scaler)

    print("\nDone!")


if __name__ == "__main__":
    main()
