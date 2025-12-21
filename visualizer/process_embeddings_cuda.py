"""
Process activations into UMAP embeddings using CUDA GPU acceleration.
Supports multi-GPU via Dask-cuML for large datasets.

Requires NVIDIA GPU with RAPIDS cuML installed.
For CPU-only processing, use process_embeddings.py instead.

Installation:
    pip install cuml-cu12 cudf-cu12 dask-cuda  # For CUDA 12
    # Or use RAPIDS conda: conda install -c rapidsai cuml dask-cuda

Usage:
    # Single GPU
    python process_embeddings_cuda.py --model imagenet_real --data_dir data

    # Multi-GPU (auto-detect)
    python process_embeddings_cuda.py --model imagenet_real --data_dir data --multi_gpu

    # Specific number of GPUs
    python process_embeddings_cuda.py --model imagenet_real --data_dir data --num_gpus 4
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import time
import os

from extract_activations import load_activations, flatten_activations

# Check for cuML availability
CUML_AVAILABLE = False
DASK_CUDA_AVAILABLE = False

try:
    from cuml.manifold import UMAP as cumlUMAP
    from cuml.preprocessing import StandardScaler as cumlStandardScaler
    import cupy as cp
    CUML_AVAILABLE = True
except ImportError:
    pass

try:
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    from cuml.dask.manifold import UMAP as DaskUMAP
    from cuml.dask.preprocessing import StandardScaler as DaskStandardScaler
    import dask.array as da
    DASK_CUDA_AVAILABLE = True
except ImportError:
    pass

if not CUML_AVAILABLE:
    print("Warning: cuML not available. Install with: pip install cuml-cu12")
    print("Falling back to CPU UMAP...")
    from umap import UMAP as cpuUMAP
    from sklearn.preprocessing import StandardScaler as cpuStandardScaler


def get_gpu_count():
    """Get number of available NVIDIA GPUs."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def load_dataset_activations(
    activation_dir: Path,
    metadata_path: Path,
    max_samples: int = None,
    batch_size: int = 500
):
    """
    Load all activations from dataset.

    Args:
        activation_dir: Directory containing activation .npz files
        metadata_path: Path to dataset_info.json
        max_samples: Optional limit on samples to load
        batch_size: Number of samples to load per batch

    Returns:
        (activation_matrix, metadata_df)
    """
    # Load metadata
    with open(metadata_path, 'r') as f:
        dataset_info = json.load(f)

    samples = dataset_info['samples'][:max_samples] if max_samples else dataset_info['samples']

    print(f"Loading {len(samples)} samples...")

    # First pass: determine activation shape
    first_sample = samples[0]

    if 'activation_path' in first_sample:
        data_root = activation_dir.parent.parent
        first_path = data_root / first_sample['activation_path']
    else:
        first_path = activation_dir / f"{first_sample['sample_id']}.npz"

    first_act, _ = load_activations(first_path)
    first_flat = flatten_activations(first_act)
    activation_dim = first_flat.shape[1]

    # Preallocate array
    activation_matrix = np.zeros((len(samples), activation_dim), dtype=np.float32)
    metadata_records = []

    # Load in batches
    valid_idx = 0
    batch_cache = {}

    for i in tqdm(range(0, len(samples), batch_size), desc="Loading batches"):
        batch_samples = samples[i:i+batch_size]

        for sample in batch_samples:
            sample_id = sample['sample_id']

            if 'batch_index' in sample:
                act_path_str = sample['activation_path']
                batch_index = sample['batch_index']
                data_root = activation_dir.parent.parent
                act_path = data_root / act_path_str
            else:
                act_path = activation_dir / f"{sample_id}.npz"
                batch_index = 0

            if not act_path.with_suffix('.npz').exists():
                print(f"Warning: Missing {act_path.with_suffix('.npz')}")
                continue

            act_path_key = str(act_path)
            if act_path_key not in batch_cache:
                activations, _ = load_activations(act_path)
                batch_cache[act_path_key] = activations
            else:
                activations = batch_cache[act_path_key]

            flat_act = flatten_activations(activations)
            activation_matrix[valid_idx] = flat_act[batch_index]
            metadata_records.append(sample)
            valid_idx += 1

    activation_matrix = activation_matrix[:valid_idx]
    metadata_df = pd.DataFrame(metadata_records)

    print(f"Loaded activations: {activation_matrix.shape}")

    # Handle NaN/inf
    nan_count = np.isnan(activation_matrix).sum()
    inf_count = np.isinf(activation_matrix).sum()

    if nan_count > 0 or inf_count > 0:
        print(f"Warning: Found {nan_count} NaN and {inf_count} inf values")
        print("Replacing with 0...")
        activation_matrix = np.nan_to_num(activation_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return activation_matrix, metadata_df


def compute_umap_single_gpu(
    activations: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    n_components: int = 2,
    random_state: int = 42,
    normalize: bool = True
):
    """
    Compute UMAP using single CUDA GPU (cuML).
    """
    print(f"\nComputing UMAP with cuML (single GPU):")
    print(f"  n_neighbors={n_neighbors}")
    print(f"  min_dist={min_dist}")
    print(f"  metric={metric}")
    print(f"  n_components={n_components}")
    print(f"  samples={activations.shape[0]}, features={activations.shape[1]}")

    start_time = time.time()

    # Transfer to GPU
    print("Transferring data to GPU...")
    activations_gpu = cp.asarray(activations, dtype=cp.float32)

    # Normalize on GPU
    if normalize:
        print("Normalizing on GPU...")
        scaler = cumlStandardScaler()
        activations_gpu = scaler.fit_transform(activations_gpu)

    # Compute UMAP on GPU
    print("Running UMAP on GPU...")
    reducer = cumlUMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=n_components,
        random_state=random_state,
        verbose=True
    )

    embeddings_gpu = reducer.fit_transform(activations_gpu)

    # Transfer back to CPU
    print("Transferring results to CPU...")
    embeddings = cp.asnumpy(embeddings_gpu)

    elapsed = time.time() - start_time
    print(f"\nUMAP completed in {elapsed:.1f}s")
    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings


def compute_umap_multi_gpu(
    activations: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    n_components: int = 2,
    random_state: int = 42,
    normalize: bool = True,
    num_gpus: int = None
):
    """
    Compute UMAP using multiple GPUs via Dask-cuML.
    """
    if num_gpus is None:
        num_gpus = get_gpu_count()

    print(f"\nComputing UMAP with Dask-cuML ({num_gpus} GPUs):")
    print(f"  n_neighbors={n_neighbors}")
    print(f"  min_dist={min_dist}")
    print(f"  metric={metric}")
    print(f"  n_components={n_components}")
    print(f"  samples={activations.shape[0]}, features={activations.shape[1]}")

    start_time = time.time()

    # Create Dask CUDA cluster
    print(f"Starting Dask CUDA cluster with {num_gpus} GPUs...")
    cluster = LocalCUDACluster(n_workers=num_gpus)
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")

    try:
        # Convert to Dask array and distribute across GPUs
        print("Distributing data across GPUs...")
        chunk_size = len(activations) // num_gpus
        dask_activations = da.from_array(activations, chunks=(chunk_size, -1))

        # Normalize with distributed scaler
        if normalize:
            print("Normalizing across GPUs...")
            scaler = DaskStandardScaler()
            dask_activations = scaler.fit_transform(dask_activations)

        # Compute UMAP across GPUs
        print("Running distributed UMAP...")
        reducer = DaskUMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state,
            verbose=True
        )

        embeddings_dask = reducer.fit_transform(dask_activations)

        # Collect results
        print("Collecting results...")
        embeddings = embeddings_dask.compute()

        # Convert from cupy if needed
        if hasattr(embeddings, 'get'):
            embeddings = embeddings.get()

    finally:
        # Cleanup
        client.close()
        cluster.close()

    elapsed = time.time() - start_time
    print(f"\nUMAP completed in {elapsed:.1f}s")
    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings


def compute_umap_cpu(
    activations: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    n_components: int = 2,
    random_state: int = 42,
    normalize: bool = True
):
    """
    Fallback CPU UMAP implementation.
    """
    print(f"\nComputing UMAP with CPU (fallback):")
    print(f"  n_neighbors={n_neighbors}")
    print(f"  min_dist={min_dist}")
    print(f"  samples={activations.shape[0]}, features={activations.shape[1]}")

    start_time = time.time()

    if normalize:
        print("Normalizing...")
        scaler = cpuStandardScaler()
        activations = scaler.fit_transform(activations)

    print("Running UMAP (this may take a while on CPU)...")
    reducer = cpuUMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=n_components,
        random_state=random_state,
        verbose=True
    )

    embeddings = reducer.fit_transform(activations)

    elapsed = time.time() - start_time
    print(f"\nUMAP completed in {elapsed:.1f}s")
    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    output_path: Path,
    umap_params: dict
):
    """
    Save UMAP embeddings + metadata to CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = metadata_df.copy()

    if embeddings.shape[1] == 2:
        df['umap_x'] = embeddings[:, 0]
        df['umap_y'] = embeddings[:, 1]
    else:
        df['umap_x'] = embeddings[:, 0]
        df['umap_y'] = embeddings[:, 1]
        df['umap_z'] = embeddings[:, 2]

    df.to_csv(output_path, index=False)
    print(f"\nSaved embeddings to {output_path}")

    param_path = output_path.with_suffix('.json')
    with open(param_path, 'w') as f:
        json.dump(umap_params, f, indent=2)
    print(f"Saved parameters to {param_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process DMD2 activations into UMAP embeddings (CUDA GPU)"
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
        "--multi_gpu",
        action="store_true",
        help="Use multiple GPUs via Dask-cuML"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)"
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU UMAP even if cuML is available"
    )

    args = parser.parse_args()

    # Determine backend
    gpu_count = get_gpu_count()
    use_multi_gpu = (args.multi_gpu or args.num_gpus is not None) and DASK_CUDA_AVAILABLE and gpu_count > 1
    use_single_gpu = CUML_AVAILABLE and not args.force_cpu and not use_multi_gpu

    if use_multi_gpu:
        num_gpus = args.num_gpus or gpu_count
        print(f"Using {num_gpus} GPUs (Dask-cuML)")
    elif use_single_gpu:
        print("Using single GPU (cuML)")
        try:
            print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
            print(f"GPU Memory: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB")
        except Exception as e:
            print(f"GPU info unavailable: {e}")
    else:
        print("Using CPU UMAP")

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
    normalize = not args.no_normalize

    if use_multi_gpu:
        embeddings = compute_umap_multi_gpu(
            activations,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            n_components=args.n_components,
            random_state=args.random_state,
            normalize=normalize,
            num_gpus=args.num_gpus
        )
        backend = f"dask-cuml-{args.num_gpus or gpu_count}gpu"
    elif use_single_gpu:
        embeddings = compute_umap_single_gpu(
            activations,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            n_components=args.n_components,
            random_state=args.random_state,
            normalize=normalize
        )
        backend = "cuml"
    else:
        embeddings = compute_umap_cpu(
            activations,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            n_components=args.n_components,
            random_state=args.random_state,
            normalize=normalize
        )
        backend = "umap-learn"

    # Save results
    output_name = f"{args.model}_umap_n{args.n_neighbors}_d{args.min_dist}.csv"
    output_path = output_dir / output_name

    umap_params = {
        "model": args.model,
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "metric": args.metric,
        "n_components": args.n_components,
        "normalize": normalize,
        "random_state": args.random_state,
        "num_samples": len(activations),
        "backend": backend
    }

    save_embeddings(embeddings, metadata_df, output_path, umap_params)

    print("\nDone!")


if __name__ == "__main__":
    main()
