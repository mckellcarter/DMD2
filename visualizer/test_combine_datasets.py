"""
Tests for dataset combination functionality.
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import json

from data_sources import combine_datasets, create_output_dirs


class TestCombineDatasets:
    """Test dataset combination functionality."""

    def _create_mock_dataset(self, path: Path, num_batches: int, samples_per_batch: int):
        """Create a mock extracted dataset structure."""
        dirs = create_output_dirs(path)

        samples = []
        sample_idx = 0

        for batch_idx in range(num_batches):
            batch_id = f"batch_{batch_idx:06d}"

            # Create activation NPZ
            act_data = {"encoder_bottleneck": np.random.randn(samples_per_batch, 1024)}
            np.savez(dirs['activation'] / f"{batch_id}.npz", **act_data)

            # Create batch metadata JSON
            batch_meta = {"batch_size": samples_per_batch, "layers": ["encoder_bottleneck"]}
            with open(dirs['activation'] / f"{batch_id}.json", 'w') as f:
                json.dump(batch_meta, f)

            # Create sample images and metadata
            for i in range(samples_per_batch):
                sample_id = f"sample_{sample_idx:06d}"

                # Create mock images
                img = Image.new('RGB', (64, 64), color='red')
                img.save(dirs['image'] / f"{sample_id}.png")
                img.save(dirs['reconstructed'] / f"{sample_id}.png")

                samples.append({
                    "sample_id": sample_id,
                    "class_label": sample_idx % 10,
                    "synset_id": "n01440764",
                    "class_name": "tench",
                    "image_path": f"images/imagenet_real/{sample_id}.png",
                    "reconstructed_path": f"images/imagenet_real_reconstructed/{sample_id}.png",
                    "activation_path": f"activations/imagenet_real/{batch_id}",
                    "batch_index": i,
                    "source": "test_source"
                })
                sample_idx += 1

        # Create dataset_info.json
        dataset_info = {
            "model_type": "imagenet_real",
            "num_samples": len(samples),
            "layers": ["encoder_bottleneck"],
            "samples": samples
        }
        with open(dirs['metadata'] / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f)

        return path

    def test_combine_two_datasets(self, tmp_path):
        """Test combining two datasets."""
        ds1 = self._create_mock_dataset(tmp_path / "dataset1", num_batches=2, samples_per_batch=3)
        ds2 = self._create_mock_dataset(tmp_path / "dataset2", num_batches=2, samples_per_batch=3)

        output_dir = tmp_path / "combined"
        metadata_path = combine_datasets([ds1, ds2], output_dir)

        assert metadata_path.exists()

        with open(metadata_path, 'r') as f:
            combined = json.load(f)

        # Should have 12 samples (2 datasets × 2 batches × 3 samples)
        assert combined['num_samples'] == 12
        assert combined['num_batches'] == 4
        assert len(combined['samples']) == 12

        # Verify sample IDs are sequential
        sample_ids = [s['sample_id'] for s in combined['samples']]
        expected = [f"sample_{i:06d}" for i in range(12)]
        assert sample_ids == expected

        # Verify activation files exist
        activation_dir = output_dir / "activations" / "imagenet_real"
        assert (activation_dir / "batch_000000.npz").exists()
        assert (activation_dir / "batch_000003.npz").exists()

    def test_combine_skip_images(self, tmp_path):
        """Test combining datasets without copying images."""
        ds1 = self._create_mock_dataset(tmp_path / "dataset1", num_batches=1, samples_per_batch=2)

        output_dir = tmp_path / "combined"
        combine_datasets([ds1], output_dir, copy_images=False)

        # Images should not exist in output
        image_dir = output_dir / "images" / "imagenet_real"
        assert len(list(image_dir.glob("*.png"))) == 0

        # But activations should
        activation_dir = output_dir / "activations" / "imagenet_real"
        assert len(list(activation_dir.glob("*.npz"))) == 1

    def test_combine_preserves_class_distribution(self, tmp_path):
        """Test that class labels are preserved after combining."""
        ds1 = self._create_mock_dataset(tmp_path / "dataset1", num_batches=1, samples_per_batch=5)
        ds2 = self._create_mock_dataset(tmp_path / "dataset2", num_batches=1, samples_per_batch=5)

        output_dir = tmp_path / "combined"
        metadata_path = combine_datasets([ds1, ds2], output_dir)

        with open(metadata_path, 'r') as f:
            combined = json.load(f)

        # Check class labels preserved
        class_labels = [s['class_label'] for s in combined['samples']]
        assert len(class_labels) == 10

    def test_combine_tracks_original_dataset(self, tmp_path):
        """Test that original dataset info is tracked."""
        ds1 = self._create_mock_dataset(tmp_path / "dataset1", num_batches=1, samples_per_batch=2)
        ds2 = self._create_mock_dataset(tmp_path / "dataset2", num_batches=1, samples_per_batch=2)

        output_dir = tmp_path / "combined"
        metadata_path = combine_datasets([ds1, ds2], output_dir)

        with open(metadata_path, 'r') as f:
            combined = json.load(f)

        # Check source datasets tracked
        assert len(combined['source_datasets']) == 2
        assert str(ds1) in combined['source_datasets']
        assert str(ds2) in combined['source_datasets']

        # Check each sample tracks original
        for sample in combined['samples']:
            assert 'original_dataset' in sample
            assert 'original_sample_id' in sample

    def test_combine_single_dataset(self, tmp_path):
        """Test combining a single dataset (copy operation)."""
        ds1 = self._create_mock_dataset(tmp_path / "dataset1", num_batches=2, samples_per_batch=3)

        output_dir = tmp_path / "combined"
        metadata_path = combine_datasets([ds1], output_dir)

        with open(metadata_path, 'r') as f:
            combined = json.load(f)

        assert combined['num_samples'] == 6
        assert combined['num_batches'] == 2

    def test_combine_with_max_samples(self, tmp_path):
        """Test limiting samples per dataset with max_samples_per_dataset."""
        # Create two datasets with 6 samples each (2 batches × 3 samples)
        ds1 = self._create_mock_dataset(tmp_path / "dataset1", num_batches=2, samples_per_batch=3)
        ds2 = self._create_mock_dataset(tmp_path / "dataset2", num_batches=2, samples_per_batch=3)

        output_dir = tmp_path / "combined"
        # Take only first 4 samples from each dataset
        metadata_path = combine_datasets([ds1, ds2], output_dir, max_samples_per_dataset=4)

        with open(metadata_path, 'r') as f:
            combined = json.load(f)

        # Should have 8 samples (4 from each dataset)
        assert combined['num_samples'] == 8
        assert len(combined['samples']) == 8

        # Verify sample IDs are sequential
        sample_ids = [s['sample_id'] for s in combined['samples']]
        expected = [f"sample_{i:06d}" for i in range(8)]
        assert sample_ids == expected

        # Verify original sample IDs show first 4 from each
        orig_ids_ds1 = [s['original_sample_id'] for s in combined['samples'][:4]]
        orig_ids_ds2 = [s['original_sample_id'] for s in combined['samples'][4:]]
        assert orig_ids_ds1 == [f"sample_{i:06d}" for i in range(4)]
        assert orig_ids_ds2 == [f"sample_{i:06d}" for i in range(4)]

    def test_combine_max_samples_only_copies_needed_batches(self, tmp_path):
        """Test that only batches containing kept samples are copied."""
        # Create dataset with 3 batches × 10 samples = 30 samples
        ds1 = self._create_mock_dataset(tmp_path / "dataset1", num_batches=3, samples_per_batch=10)

        output_dir = tmp_path / "combined"
        # Take only first 15 samples (should only need batches 0 and 1)
        metadata_path = combine_datasets([ds1], output_dir, max_samples_per_dataset=15)

        with open(metadata_path, 'r') as f:
            combined = json.load(f)

        assert combined['num_samples'] == 15
        # Should only have 2 batches (batch 2 not needed)
        assert combined['num_batches'] == 2

        # Verify only 2 batch files exist
        activation_dir = output_dir / "activations" / "imagenet_real"
        batch_files = list(activation_dir.glob("batch_*.npz"))
        assert len(batch_files) == 2

    def test_combine_max_samples_exceeds_dataset_size(self, tmp_path):
        """Test max_samples larger than dataset just takes all samples."""
        ds1 = self._create_mock_dataset(tmp_path / "dataset1", num_batches=1, samples_per_batch=5)

        output_dir = tmp_path / "combined"
        # Request 100 samples but dataset only has 5
        metadata_path = combine_datasets([ds1], output_dir, max_samples_per_dataset=100)

        with open(metadata_path, 'r') as f:
            combined = json.load(f)

        # Should just have all 5 samples
        assert combined['num_samples'] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
