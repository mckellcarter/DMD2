"""
Unit tests for visualization_app module.
Tests JSON auto-detection of adapter/checkpoint.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd


class TestAdapterAutoDetection:
    """Tests for adapter/checkpoint auto-detection from embeddings JSON."""

    def test_auto_detects_adapter_from_json(self, tmp_path):
        """Verify adapter is loaded from JSON when not specified via CLI."""
        # Create mock embeddings CSV and JSON
        csv_path = tmp_path / "embeddings.csv"
        json_path = tmp_path / "embeddings.json"

        pd.DataFrame({
            'sample_id': ['s0', 's1'],
            'umap_x': [0.1, 0.2],
            'umap_y': [0.3, 0.4]
        }).to_csv(csv_path, index=False)

        with open(json_path, 'w') as f:
            json.dump({
                "adapter": "edm-imagenet-64",
                "checkpoint": "checkpoints/edm.pkl",
                "n_neighbors": 15
            }, f)

        # Import and patch to avoid full initialization
        from visualization_app import DMD2Visualizer

        with patch.object(DMD2Visualizer, 'load_class_labels'):
            with patch.object(DMD2Visualizer, 'build_layout'):
                with patch.object(DMD2Visualizer, 'register_callbacks'):
                    with patch('visualization_app.dash.Dash'):
                        viz = DMD2Visualizer(
                            data_dir=tmp_path,
                            embeddings_path=csv_path,
                            adapter_name=None,  # Not specified
                            checkpoint_path=None  # Not specified
                        )

        assert viz.adapter_name == "edm-imagenet-64"
        assert viz.checkpoint_path == "checkpoints/edm.pkl"

    def test_cli_adapter_overrides_json(self, tmp_path):
        """Verify CLI adapter takes precedence over JSON."""
        csv_path = tmp_path / "embeddings.csv"
        json_path = tmp_path / "embeddings.json"

        pd.DataFrame({
            'sample_id': ['s0'],
            'umap_x': [0.1],
            'umap_y': [0.2]
        }).to_csv(csv_path, index=False)

        with open(json_path, 'w') as f:
            json.dump({
                "adapter": "edm-imagenet-64",
                "checkpoint": "checkpoints/edm.pkl"
            }, f)

        from visualization_app import DMD2Visualizer

        with patch.object(DMD2Visualizer, 'load_class_labels'):
            with patch.object(DMD2Visualizer, 'build_layout'):
                with patch.object(DMD2Visualizer, 'register_callbacks'):
                    with patch('visualization_app.dash.Dash'):
                        viz = DMD2Visualizer(
                            data_dir=tmp_path,
                            embeddings_path=csv_path,
                            adapter_name="dmd2-imagenet-64",  # CLI override
                            checkpoint_path="/other/path.pt"  # CLI override
                        )

        # CLI values should be used, not JSON
        assert viz.adapter_name == "dmd2-imagenet-64"
        assert viz.checkpoint_path == "/other/path.pt"

    def test_defaults_to_dmd2_when_json_missing_adapter(self, tmp_path):
        """Verify default adapter when JSON has no adapter field."""
        csv_path = tmp_path / "embeddings.csv"
        json_path = tmp_path / "embeddings.json"

        pd.DataFrame({
            'sample_id': ['s0'],
            'umap_x': [0.1],
            'umap_y': [0.2]
        }).to_csv(csv_path, index=False)

        # Old format JSON without adapter
        with open(json_path, 'w') as f:
            json.dump({"n_neighbors": 15, "min_dist": 0.1}, f)

        from visualization_app import DMD2Visualizer

        with patch.object(DMD2Visualizer, 'load_class_labels'):
            with patch.object(DMD2Visualizer, 'build_layout'):
                with patch.object(DMD2Visualizer, 'register_callbacks'):
                    with patch('visualization_app.dash.Dash'):
                        viz = DMD2Visualizer(
                            data_dir=tmp_path,
                            embeddings_path=csv_path,
                            adapter_name=None
                        )

        # Should fall back to default
        assert viz.adapter_name == "dmd2-imagenet-64"

    def test_checkpoint_none_when_json_missing(self, tmp_path):
        """Verify checkpoint stays None when not in JSON or CLI."""
        csv_path = tmp_path / "embeddings.csv"
        json_path = tmp_path / "embeddings.json"

        pd.DataFrame({
            'sample_id': ['s0'],
            'umap_x': [0.1],
            'umap_y': [0.2]
        }).to_csv(csv_path, index=False)

        with open(json_path, 'w') as f:
            json.dump({"adapter": "edm-imagenet-64"}, f)  # No checkpoint

        from visualization_app import DMD2Visualizer

        with patch.object(DMD2Visualizer, 'load_class_labels'):
            with patch.object(DMD2Visualizer, 'build_layout'):
                with patch.object(DMD2Visualizer, 'register_callbacks'):
                    with patch('visualization_app.dash.Dash'):
                        viz = DMD2Visualizer(
                            data_dir=tmp_path,
                            embeddings_path=csv_path,
                            adapter_name=None,
                            checkpoint_path=None
                        )

        assert viz.adapter_name == "edm-imagenet-64"
        assert viz.checkpoint_path is None


class TestUmapParamsLoading:
    """Tests for UMAP params JSON loading."""

    def test_loads_layers_from_json(self, tmp_path):
        """Verify layers are loaded from UMAP params JSON."""
        csv_path = tmp_path / "embeddings.csv"
        json_path = tmp_path / "embeddings.json"

        pd.DataFrame({
            'sample_id': ['s0'],
            'umap_x': [0.1],
            'umap_y': [0.2]
        }).to_csv(csv_path, index=False)

        with open(json_path, 'w') as f:
            json.dump({
                "adapter": "edm-imagenet-64",
                "layers": ["encoder_bottleneck", "midblock", "decoder_block_0"],
                "n_neighbors": 20
            }, f)

        from visualization_app import DMD2Visualizer

        with patch.object(DMD2Visualizer, 'load_class_labels'):
            with patch.object(DMD2Visualizer, 'build_layout'):
                with patch.object(DMD2Visualizer, 'register_callbacks'):
                    with patch('visualization_app.dash.Dash'):
                        viz = DMD2Visualizer(
                            data_dir=tmp_path,
                            embeddings_path=csv_path
                        )

        assert viz.umap_params.get("layers") == ["encoder_bottleneck", "midblock", "decoder_block_0"]
