"""
Data source abstractions for ImageNet activation extraction.
Supports LMDB, NPZ, and JPEG directory formats with unified interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import shutil

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# LMDB support
import lmdb
from main.utils import retrieve_row_from_lmdb, get_array_shape_from_lmdb

# Label remapping
from class_mapping import remap_imagenet64_labels_to_standard


class ImageNetDataSource(ABC):
    """Abstract interface for ImageNet data loading."""

    @abstractmethod
    def scan_samples(
        self,
        target_classes: Optional[List[int]],
        samples_per_class: int,
        num_samples: int
    ) -> List[int]:
        """
        Return indices of samples matching class criteria.

        Args:
            target_classes: List of class IDs to sample from, or None for all classes
            samples_per_class: Max samples per class for balanced sampling
            num_samples: Total samples to collect

        Returns:
            List of sample indices to process
        """

    @abstractmethod
    def load_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load a batch of samples.

        Args:
            indices: List of sample indices to load

        Returns:
            Tuple of (images, labels, source_paths) where:
                - images: uint8 array of shape (B, 3, 64, 64)
                - labels: int64 array of shape (B,) in source ordering
                - source_paths: List of source identifiers
        """

    @abstractmethod
    def get_standard_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Convert source labels to standard ImageNet-1K ordering (0-999).

        Args:
            labels: Array of labels in source ordering

        Returns:
            Array of labels in standard ImageNet ordering
        """

    @abstractmethod
    def close(self):
        """Release resources (LMDB env, file handles, etc)."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class LMDBDataSource(ImageNetDataSource):
    """LMDB dataset - labels already in standard ImageNet ordering."""

    def __init__(self, lmdb_path: Path):
        self.lmdb_path = Path(lmdb_path)
        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        self.image_shape = get_array_shape_from_lmdb(self.env, 'images')
        self.label_shape = get_array_shape_from_lmdb(self.env, 'labels')
        self.total_samples = self.image_shape[0]
        print(f"LMDB contains {self.total_samples:,} samples, image shape: {self.image_shape[1:]}")

    def scan_samples(
        self,
        target_classes: Optional[List[int]],
        samples_per_class: int,
        num_samples: int
    ) -> List[int]:
        if target_classes is None:
            target_classes = list(range(1000))

        target_set = set(target_classes)
        class_counts = {c: 0 for c in target_classes}
        selected_indices = []

        print(f"Scanning LMDB for {len(target_classes)} classes...")
        for idx in tqdm(range(self.total_samples), desc="Scanning labels"):
            label = retrieve_row_from_lmdb(
                self.env, "labels", np.int64, self.label_shape[1:], idx
            )
            label_int = int(label.item()) if hasattr(label, 'item') else int(label)

            if label_int in target_set and class_counts[label_int] < samples_per_class:
                selected_indices.append(idx)
                class_counts[label_int] += 1

                if len(selected_indices) >= num_samples:
                    break

        print(f"Collected {len(selected_indices):,} samples")
        return selected_indices

    def load_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        batch_size = len(indices)
        images = np.zeros((batch_size, 3, 64, 64), dtype=np.uint8)
        labels = np.zeros(batch_size, dtype=np.int64)
        paths = []

        for i, idx in enumerate(indices):
            images[i] = retrieve_row_from_lmdb(
                self.env, "images", np.uint8, self.image_shape[1:], idx
            )
            label = retrieve_row_from_lmdb(
                self.env, "labels", np.int64, self.label_shape[1:], idx
            )
            labels[i] = int(label.item()) if hasattr(label, 'item') else int(label)
            paths.append(f"lmdb_idx_{idx}")

        return images, labels, paths

    def get_standard_labels(self, labels: np.ndarray) -> np.ndarray:
        # LMDB already has standard labels
        return labels

    def close(self):
        self.env.close()


class NPZDataSource(ImageNetDataSource):
    """ImageNet64 NPZ batches - requires ImageNet64 -> Standard label remapping."""

    def __init__(self, npz_dir: Path):
        self.npz_dir = Path(npz_dir)
        # Sort numerically by batch number
        self.npz_files = sorted(
            list(self.npz_dir.glob('*.npz')),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        if not self.npz_files:
            raise FileNotFoundError(f"No NPZ files found in {npz_dir}")

        # Count samples and build index mapping
        self.npz_batch_sizes = []
        self.idx_to_npz = []  # (file_idx, within_idx)
        total = 0

        for file_idx, npz_file in enumerate(self.npz_files):
            data = np.load(npz_file)
            batch_size = data['data'].shape[0]
            self.npz_batch_sizes.append(batch_size)
            for within_idx in range(batch_size):
                self.idx_to_npz.append((file_idx, within_idx))
            total += batch_size

        self.total_samples = total
        print(f"Found {len(self.npz_files)} NPZ files with {self.total_samples:,} total samples")

    def scan_samples(
        self,
        target_classes: Optional[List[int]],
        samples_per_class: int,
        num_samples: int
    ) -> List[int]:
        if target_classes is None:
            target_classes = list(range(1000))

        target_set = set(target_classes)
        class_counts = {c: 0 for c in target_classes}
        selected_indices = []
        global_idx = 0

        for npz_file in self.npz_files:
            if len(selected_indices) >= num_samples:
                break

            data = np.load(npz_file)
            labels_0indexed = data['labels'] - 1  # Convert to 0-indexed

            for label in labels_0indexed:
                if label in target_set and class_counts[label] < samples_per_class:
                    selected_indices.append(global_idx)
                    class_counts[label] += 1
                    if len(selected_indices) >= num_samples:
                        break
                global_idx += 1

        print(f"Collected {len(selected_indices):,} samples")
        return selected_indices

    def load_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        batch_size = len(indices)
        images = np.zeros((batch_size, 3, 64, 64), dtype=np.uint8)
        labels = np.zeros(batch_size, dtype=np.int64)
        paths = []

        # Group by NPZ file for efficient loading
        npz_groups: Dict[int, List[Tuple[int, int]]] = {}
        for local_idx, global_idx in enumerate(indices):
            file_idx, within_idx = self.idx_to_npz[global_idx]
            if file_idx not in npz_groups:
                npz_groups[file_idx] = []
            npz_groups[file_idx].append((local_idx, within_idx))

        for npz_idx, items in npz_groups.items():
            npz_file = self.npz_files[npz_idx]
            data = np.load(npz_file)
            images_flat = data['data']
            labels_1indexed = data['labels']

            for local_idx, file_within_idx in items:
                images[local_idx] = images_flat[file_within_idx].reshape(3, 64, 64)
                labels[local_idx] = labels_1indexed[file_within_idx] - 1  # 0-indexed

        for global_idx in indices:
            paths.append(f"npz_sample_{global_idx}")

        return images, labels, paths

    def get_standard_labels(self, labels: np.ndarray) -> np.ndarray:
        # NPZ uses ImageNet64 ordering, needs remapping
        return remap_imagenet64_labels_to_standard(labels)

    def close(self):
        pass  # No persistent resources


class JPEGDataSource(ImageNetDataSource):
    """Directory of JPEG images organized by synset."""

    def __init__(self, imagenet_dir: Path, split: str, class_labels_map: Dict):
        self.imagenet_dir = Path(imagenet_dir)
        self.split = split
        self.split_dir = self.imagenet_dir / split

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"ImageNet split directory not found: {self.split_dir}\n"
                f"Expected structure: {self.imagenet_dir}/{split}/n01440764/*.JPEG"
            )

        # Build synset -> class_id mapping
        self.synset_to_class = {
            v[0]: int(k) for k, v in class_labels_map.items()
        }

        # Collect image paths
        extensions = ['.JPEG', '.jpg', '.png']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(list(self.split_dir.rglob(f'*{ext}')))

        self.total_samples = len(self.image_paths)
        print(f"Found {self.total_samples:,} images in {self.split_dir}")

        if self.total_samples == 0:
            raise ValueError(f"No images found in {self.split_dir}")

    def scan_samples(
        self,
        target_classes: Optional[List[int]],
        samples_per_class: int,
        num_samples: int
    ) -> List[int]:
        # For JPEG, we just shuffle and return indices
        # Class filtering happens at load time via synset lookup
        indices = list(range(len(self.image_paths)))
        np.random.shuffle(indices)
        return indices[:num_samples]

    def load_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        batch_size = len(indices)
        images = np.zeros((batch_size, 3, 64, 64), dtype=np.uint8)
        labels = np.zeros(batch_size, dtype=np.int64)
        paths = []

        for i, idx in enumerate(indices):
            img_path = self.image_paths[idx]

            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            img_array = np.array(img).astype(np.uint8)
            images[i] = img_array.transpose(2, 0, 1)  # HWC -> CHW

            # Get class from synset (parent directory)
            synset_id = img_path.parent.name
            if synset_id in self.synset_to_class:
                labels[i] = self.synset_to_class[synset_id]
            else:
                print(f"Warning: Unknown synset {synset_id} for {img_path}")
                labels[i] = -1

            paths.append(str(img_path))

        return images, labels, paths

    def get_standard_labels(self, labels: np.ndarray) -> np.ndarray:
        # JPEG labels are already standard (from synset lookup)
        return labels

    def close(self):
        pass


class BatchProcessor:
    """Unified batch processing: tensor conversion, forward pass, saving."""

    def __init__(
        self,
        extractor,
        adapter,
        output_dirs: Dict[str, Path],
        class_labels_map: Dict,
        device: str,
        conditioning_sigma: float,
        layers: List[str],
        source_name: str = "imagenet_real"
    ):
        self.extractor = extractor
        self.adapter = adapter
        self.output_dirs = output_dirs
        self.class_labels_map = class_labels_map
        self.device = device
        self.conditioning_sigma = conditioning_sigma
        self.layers = layers
        self.source_name = source_name

    def process_batch(
        self,
        images_uint8: np.ndarray,
        labels_standard: np.ndarray,
        source_paths: List[str],
        batch_idx: int,
        sample_idx_start: int
    ) -> Tuple[int, List[Dict]]:
        """
        Process a batch: extract activations, save outputs.

        Returns:
            Tuple of (next_sample_idx, batch_metadata_records)
        """
        batch_size = len(images_uint8)

        # Convert to tensor and normalize to [-1, 1]
        batch_tensor = torch.from_numpy(images_uint8).float().to(self.device)
        batch_tensor = (batch_tensor / 127.5) - 1.0

        # Create one-hot labels
        labels_tensor = torch.from_numpy(labels_standard).long().to(self.device)
        one_hot_labels = torch.eye(1000, device=self.device)[labels_tensor]

        # Get class metadata
        synsets = []
        class_names = []
        for label_id in labels_standard:
            label_str = str(int(label_id))
            if label_str in self.class_labels_map:
                synset_id, class_name = self.class_labels_map[label_str]
            else:
                synset_id = f"unknown_{label_id}"
                class_name = "unknown"
            synsets.append(synset_id)
            class_names.append(class_name)

        # Extract activations via forward pass
        self.extractor.clear_activations()
        with torch.no_grad():
            sigma = torch.ones(batch_size, device=self.device) * self.conditioning_sigma
            reconstructed = self.adapter.forward(
                batch_tensor * self.conditioning_sigma,
                sigma,
                one_hot_labels
            )

        # Convert reconstructed to uint8
        reconstructed_uint8 = (
            ((reconstructed + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        )
        reconstructed_uint8 = reconstructed_uint8.permute(0, 2, 3, 1).cpu().numpy()

        # Get activations
        activations = self.extractor.get_activations()

        # Save activations
        batch_id = f"batch_{batch_idx:06d}"
        batch_act_path = self.output_dirs['activation'] / batch_id

        activation_dict = {}
        for name, activation in activations.items():
            if len(activation.shape) == 4:
                activation_dict[name] = activation.reshape(batch_size, -1).cpu().numpy()
            else:
                activation_dict[name] = activation.cpu().numpy()

        np.savez_compressed(str(batch_act_path.with_suffix('.npz')), **activation_dict)

        # Save batch metadata JSON
        batch_samples_meta = []
        for i in range(batch_size):
            batch_samples_meta.append({
                "batch_index": i,
                "class_id": int(labels_standard[i]),
                "synset_id": synsets[i],
                "class_name": class_names[i],
                "original_path": source_paths[i]
            })

        batch_metadata = {
            "batch_size": batch_size,
            "layers": self.layers,
            "samples": batch_samples_meta
        }

        with open(batch_act_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(batch_metadata, f, indent=2)

        # Save images and build per-sample metadata
        metadata_records = []
        sample_idx = sample_idx_start

        for i in range(batch_size):
            sample_id = f"sample_{sample_idx:06d}"

            # Save original image
            img_path = self.output_dirs['image'] / f"{sample_id}.png"
            img_np = images_uint8[i].transpose(1, 2, 0)  # CHW -> HWC
            Image.fromarray(img_np).save(img_path)

            # Save reconstructed image
            recon_path = self.output_dirs['reconstructed'] / f"{sample_id}.png"
            Image.fromarray(reconstructed_uint8[i]).save(recon_path)

            # Build metadata record
            metadata_records.append({
                "sample_id": sample_id,
                "class_label": int(labels_standard[i]),
                "synset_id": synsets[i],
                "class_name": class_names[i],
                "image_path": str(img_path.relative_to(self.output_dirs['root'])),
                "reconstructed_path": str(recon_path.relative_to(self.output_dirs['root'])),
                "activation_path": str(batch_act_path.relative_to(self.output_dirs['root'])),
                "batch_index": i,
                "original_path": source_paths[i],
                "source": self.source_name,
                "conditioning_sigma": self.conditioning_sigma
            })

            sample_idx += 1

        return sample_idx, metadata_records


def create_data_source(
    lmdb_path: Optional[Path] = None,
    npz_dir: Optional[Path] = None,
    imagenet_dir: Optional[Path] = None,
    split: str = "train",
    class_labels_map: Optional[Dict] = None
) -> ImageNetDataSource:
    """
    Factory to create appropriate data source.

    Args:
        lmdb_path: Path to LMDB dataset
        npz_dir: Directory containing NPZ batch files
        imagenet_dir: Root directory of ImageNet JPEG dataset
        split: Dataset split for JPEG format
        class_labels_map: Class labels map (required for JPEG format)

    Returns:
        ImageNetDataSource implementation
    """
    if lmdb_path is not None:
        return LMDBDataSource(lmdb_path)
    if npz_dir is not None:
        return NPZDataSource(npz_dir)
    if imagenet_dir is not None:
        if class_labels_map is None:
            raise ValueError("class_labels_map required for JPEG format")
        return JPEGDataSource(imagenet_dir, split, class_labels_map)
    raise ValueError("Must provide one of: lmdb_path, npz_dir, imagenet_dir")


def load_class_labels_map(labels_path: Optional[Path] = None) -> Dict:
    """Load ImageNet class labels map."""
    if labels_path is None:
        labels_path = Path(__file__).parent / "data" / "imagenet_standard_class_index.json"
    with open(labels_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_output_dirs(output_dir: Path) -> Dict[str, Path]:
    """Create output directory structure."""
    dirs = {
        'root': output_dir,
        'image': output_dir / "images" / "imagenet_real",
        'reconstructed': output_dir / "images" / "imagenet_real_reconstructed",
        'activation': output_dir / "activations" / "imagenet_real",
        'metadata': output_dir / "metadata" / "imagenet_real"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def combine_datasets(
    dataset_paths: List[Path],
    output_dir: Path,
    copy_images: bool = True,
    max_samples_per_dataset: Optional[int] = None
) -> Path:
    """
    Combine multiple extracted activation datasets into one.

    Keeps original batch NPZ files, renumbers them sequentially.
    Updates sample IDs and metadata paths accordingly.

    Args:
        dataset_paths: List of paths to dataset root directories
            (each should contain metadata/imagenet_real/dataset_info.json)
        output_dir: Output directory for combined dataset
        copy_images: If True, copy image files. If False, only combine activations/metadata.
        max_samples_per_dataset: If set, take only first N samples from each dataset.

    Returns:
        Path to combined dataset_info.json
    """
    output_dirs = create_output_dirs(output_dir)

    combined_samples = []
    batch_idx = 0
    sample_idx = 0
    combined_layers = None
    combined_sources = set()

    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        metadata_path = dataset_path / "metadata" / "imagenet_real" / "dataset_info.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)

        # Track layers (should be consistent across datasets)
        if combined_layers is None:
            combined_layers = dataset_info.get('layers', [])

        # Get samples, optionally limited
        dataset_samples = dataset_info.get('samples', [])
        if max_samples_per_dataset is not None:
            dataset_samples = dataset_samples[:max_samples_per_dataset]

        # Track sources
        for sample in dataset_samples:
            combined_sources.add(sample.get('source', 'unknown'))

        # Find which batches we need (only those containing samples we're keeping)
        needed_batches = set()
        for sample in dataset_samples:
            old_act_path = sample.get('activation_path', '')
            if old_act_path:
                old_batch_id = Path(old_act_path).stem
                needed_batches.add(old_batch_id)

        # Find activation batches in this dataset
        activation_dir = dataset_path / "activations" / "imagenet_real"
        batch_files = sorted(activation_dir.glob("batch_*.npz"))

        # Map old batch paths to new batch indices (only for needed batches)
        old_to_new_batch = {}

        for old_batch_file in batch_files:
            old_batch_id = old_batch_file.stem  # e.g., "batch_000005"

            # Skip batches we don't need
            if old_batch_id not in needed_batches:
                continue

            new_batch_id = f"batch_{batch_idx:06d}"

            # Copy activation NPZ
            new_act_path = output_dirs['activation'] / f"{new_batch_id}.npz"
            shutil.copy2(old_batch_file, new_act_path)

            # Copy batch metadata JSON if exists
            old_json = old_batch_file.with_suffix('.json')
            if old_json.exists():
                new_json = output_dirs['activation'] / f"{new_batch_id}.json"
                shutil.copy2(old_json, new_json)

            old_to_new_batch[old_batch_id] = new_batch_id
            batch_idx += 1

        # Process samples from this dataset
        for sample in dataset_samples:
            old_sample_id = sample['sample_id']
            new_sample_id = f"sample_{sample_idx:06d}"

            # Determine new batch path
            old_act_path = sample.get('activation_path', '')
            old_batch_id = Path(old_act_path).stem if old_act_path else None
            new_batch_id = old_to_new_batch.get(old_batch_id, old_batch_id)

            # Copy images if requested
            new_image_path = None
            new_recon_path = None

            if copy_images:
                # Copy original image
                old_img = dataset_path / sample.get('image_path', '')
                if old_img.exists():
                    ext = old_img.suffix
                    new_img = output_dirs['image'] / f"{new_sample_id}{ext}"
                    shutil.copy2(old_img, new_img)
                    new_image_path = str(new_img.relative_to(output_dir))

                # Copy reconstructed image
                old_recon = dataset_path / sample.get('reconstructed_path', '')
                if old_recon.exists():
                    ext = old_recon.suffix
                    new_recon = output_dirs['reconstructed'] / f"{new_sample_id}{ext}"
                    shutil.copy2(old_recon, new_recon)
                    new_recon_path = str(new_recon.relative_to(output_dir))

            # Build new sample metadata
            new_sample = {
                "sample_id": new_sample_id,
                "class_label": sample.get('class_label'),
                "synset_id": sample.get('synset_id'),
                "class_name": sample.get('class_name'),
                "image_path": new_image_path or sample.get('image_path'),
                "reconstructed_path": new_recon_path or sample.get('reconstructed_path'),
                "activation_path": f"activations/imagenet_real/{new_batch_id}",
                "batch_index": sample.get('batch_index'),
                "original_path": sample.get('original_path'),
                "source": sample.get('source'),
                "conditioning_sigma": sample.get('conditioning_sigma'),
                "original_dataset": str(dataset_path),
                "original_sample_id": old_sample_id
            }
            combined_samples.append(new_sample)
            sample_idx += 1

    # Save combined metadata
    combined_info = {
        "model_type": "imagenet_real_combined",
        "num_samples": len(combined_samples),
        "num_batches": batch_idx,
        "layers": combined_layers,
        "sources": list(combined_sources),
        "source_datasets": [str(p) for p in dataset_paths],
        "samples": combined_samples
    }

    metadata_path = output_dirs['metadata'] / "dataset_info.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(combined_info, f, indent=2)

    print(f"Combined {len(dataset_paths)} datasets:")
    print(f"  Total samples: {len(combined_samples)}")
    print(f"  Total batches: {batch_idx}")
    print(f"  Sources: {combined_sources}")
    print(f"  Output: {metadata_path}")

    return metadata_path
