"""
Unit tests for process_embeddings module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

from process_embeddings import load_dataset_activations, compute_umap, save_embeddings


class TestLoadDatasetActivations:
    """Tests for load_dataset_activations function."""

    def test_returns_three_values(self, tmp_path):
        """Verify function returns (activations, metadata_df, dataset_info)."""
        # Create mock dataset structure
        activation_dir = tmp_path / "activations" / "test_model"
        metadata_dir = tmp_path / "metadata" / "test_model"
        activation_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)

        # Create mock activation file
        activations = {
            'encoder_bottleneck': np.random.randn(2, 128).astype(np.float32),
            'midblock': np.random.randn(2, 256).astype(np.float32)
        }
        np.savez(activation_dir / "batch_000000.npz", **activations)

        # Create mock dataset_info.json
        dataset_info = {
            "model_type": "test_model",
            "adapter": "test-adapter",
            "checkpoint": "/path/to/checkpoint.pkl",
            "layers": ["encoder_bottleneck", "midblock"],
            "num_samples": 2,
            "samples": [
                {"sample_id": "sample_000000", "activation_path": "activations/test_model/batch_000000", "batch_index": 0},
                {"sample_id": "sample_000001", "activation_path": "activations/test_model/batch_000000", "batch_index": 1}
            ]
        }
        with open(metadata_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f)

        # Call function
        result = load_dataset_activations(activation_dir, metadata_dir / "dataset_info.json")

        # Verify 3 return values
        assert len(result) == 3
        act_matrix, metadata_df, info = result

        # Verify types
        assert isinstance(act_matrix, np.ndarray)
        assert isinstance(metadata_df, pd.DataFrame)
        assert isinstance(info, dict)

    def test_dataset_info_contains_adapter(self, tmp_path):
        """Verify dataset_info dict contains adapter field."""
        # Create mock dataset
        activation_dir = tmp_path / "activations" / "test_model"
        metadata_dir = tmp_path / "metadata" / "test_model"
        activation_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)

        activations = {'layer1': np.random.randn(1, 64).astype(np.float32)}
        np.savez(activation_dir / "batch_000000.npz", **activations)

        dataset_info = {
            "adapter": "edm-imagenet-64",
            "checkpoint": "checkpoints/model.pkl",
            "layers": ["layer1"],
            "samples": [
                {"sample_id": "s0", "activation_path": "activations/test_model/batch_000000", "batch_index": 0}
            ]
        }
        with open(metadata_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f)

        _, _, info = load_dataset_activations(activation_dir, metadata_dir / "dataset_info.json")

        assert info.get("adapter") == "edm-imagenet-64"
        assert info.get("checkpoint") == "checkpoints/model.pkl"

    def test_dataset_info_missing_adapter_returns_none(self, tmp_path):
        """Verify missing adapter field returns None via .get()."""
        activation_dir = tmp_path / "activations" / "test_model"
        metadata_dir = tmp_path / "metadata" / "test_model"
        activation_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)

        activations = {'layer1': np.random.randn(1, 64).astype(np.float32)}
        np.savez(activation_dir / "batch_000000.npz", **activations)

        # Old format without adapter/checkpoint
        dataset_info = {
            "layers": ["layer1"],
            "samples": [
                {"sample_id": "s0", "activation_path": "activations/test_model/batch_000000", "batch_index": 0}
            ]
        }
        with open(metadata_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f)

        _, _, info = load_dataset_activations(activation_dir, metadata_dir / "dataset_info.json")

        assert info.get("adapter") is None
        assert info.get("checkpoint") is None


class TestSaveEmbeddings:
    """Tests for save_embeddings including adapter/checkpoint propagation."""

    def test_saves_adapter_and_checkpoint_to_json(self, tmp_path):
        """Verify adapter and checkpoint are saved to UMAP params JSON."""
        embeddings = np.random.randn(10, 2).astype(np.float32)
        metadata_df = pd.DataFrame({
            'sample_id': [f's{i}' for i in range(10)],
            'class_label': np.random.randint(0, 10, 10)
        })
        output_path = tmp_path / "embeddings.csv"

        umap_params = {
            "model": "imagenet_real",
            "adapter": "edm-imagenet-64",
            "checkpoint": "checkpoints/edm.pkl",
            "layers": ["encoder_bottleneck", "midblock"],
            "n_neighbors": 15,
            "min_dist": 0.1
        }

        save_embeddings(embeddings, metadata_df, output_path, umap_params)

        # Verify JSON was saved with adapter/checkpoint
        json_path = output_path.with_suffix('.json')
        assert json_path.exists()

        with open(json_path) as f:
            saved_params = json.load(f)

        assert saved_params.get("adapter") == "edm-imagenet-64"
        assert saved_params.get("checkpoint") == "checkpoints/edm.pkl"
        assert saved_params.get("layers") == ["encoder_bottleneck", "midblock"]

    def test_saves_none_adapter_when_not_provided(self, tmp_path):
        """Verify None values are preserved in JSON."""
        embeddings = np.random.randn(5, 2).astype(np.float32)
        metadata_df = pd.DataFrame({'sample_id': [f's{i}' for i in range(5)]})
        output_path = tmp_path / "embeddings.csv"

        umap_params = {
            "model": "test",
            "adapter": None,
            "checkpoint": None,
            "n_neighbors": 10
        }

        save_embeddings(embeddings, metadata_df, output_path, umap_params)

        with open(output_path.with_suffix('.json')) as f:
            saved_params = json.load(f)

        assert saved_params.get("adapter") is None
        assert saved_params.get("checkpoint") is None


class TestComputeUmap:
    """Tests for UMAP computation."""

    def test_returns_embeddings_reducer_scaler(self):
        """Verify compute_umap returns (embeddings, reducer, scaler)."""
        activations = np.random.randn(50, 100).astype(np.float32)

        result = compute_umap(activations, n_neighbors=5, min_dist=0.1)

        assert len(result) == 3
        embeddings, reducer, scaler = result

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (50, 2)
        assert reducer is not None
        assert scaler is not None

    def test_normalize_false_returns_none_scaler(self):
        """Verify scaler is None when normalize=False."""
        activations = np.random.randn(50, 100).astype(np.float32)

        _, _, scaler = compute_umap(activations, n_neighbors=5, normalize=False)

        assert scaler is None
