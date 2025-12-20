"""
ImageNet class mapping utilities.

The ImageNet-64 NPZ dataset uses a different class ordering than standard ImageNet-1K.
This module provides utilities to convert between the two orderings.

Example:
    ImageNet64 index 0 = kit_fox (n02119789)
    Standard ImageNet index 0 = tench (n01440764)

    ImageNet64 index 448 = tench (n01440764) -> Standard index 0
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union

# Module-level cache for mappings
_imagenet64_to_standard: Optional[Dict[int, int]] = None
_standard_to_imagenet64: Optional[Dict[int, int]] = None


def get_data_dir() -> Path:
    """Get the data directory containing class label JSON files."""
    return Path(__file__).parent / "data"


def load_imagenet64_to_standard_mapping(
    imagenet64_path: Optional[Path] = None,
    standard_path: Optional[Path] = None
) -> Dict[int, int]:
    """
    Load or create mapping from ImageNet64 indices to standard ImageNet indices.

    Args:
        imagenet64_path: Path to imagenet64_class_labels.json
        standard_path: Path to imagenet_standard_class_index.json

    Returns:
        Dict mapping ImageNet64 index (0-999) to standard ImageNet index (0-999)
    """
    global _imagenet64_to_standard

    if _imagenet64_to_standard is not None:
        return _imagenet64_to_standard

    data_dir = get_data_dir()

    if imagenet64_path is None:
        imagenet64_path = data_dir / "imagenet64_class_labels.json"
    if standard_path is None:
        standard_path = data_dir / "imagenet_standard_class_index.json"

    # Load ImageNet64 labels: {idx: [synset, name]}
    with open(imagenet64_path, 'r') as f:
        imagenet64_labels = json.load(f)

    # Load standard ImageNet labels: {idx: [synset, name]}
    with open(standard_path, 'r') as f:
        standard_labels = json.load(f)

    # Build synset -> standard index mapping
    synset_to_standard = {v[0]: int(k) for k, v in standard_labels.items()}

    # Build ImageNet64 -> Standard mapping
    _imagenet64_to_standard = {}
    for idx64_str, (synset, name) in imagenet64_labels.items():
        idx64 = int(idx64_str)
        if synset in synset_to_standard:
            _imagenet64_to_standard[idx64] = synset_to_standard[synset]
        else:
            print(f"Warning: Synset {synset} ({name}) not found in standard ImageNet")

    print(f"Loaded ImageNet64->Standard mapping for {len(_imagenet64_to_standard)} classes")

    # Verify with known examples
    # ImageNet64[0] = kit_fox (n02119789) -> Standard[278]
    # ImageNet64[448] = tench (n01440764) -> Standard[0]
    if 0 in _imagenet64_to_standard:
        print(f"  Verify: ImageNet64[0] -> Standard[{_imagenet64_to_standard[0]}] (kit_fox->278 expected)")
    if 448 in _imagenet64_to_standard:
        print(f"  Verify: ImageNet64[448] -> Standard[{_imagenet64_to_standard[448]}] (tench->0 expected)")

    return _imagenet64_to_standard


def load_standard_to_imagenet64_mapping(
    imagenet64_path: Optional[Path] = None,
    standard_path: Optional[Path] = None
) -> Dict[int, int]:
    """
    Load or create mapping from standard ImageNet indices to ImageNet64 indices.

    Returns:
        Dict mapping standard ImageNet index (0-999) to ImageNet64 index (0-999)
    """
    global _standard_to_imagenet64

    if _standard_to_imagenet64 is not None:
        return _standard_to_imagenet64

    # Get forward mapping and invert it
    forward = load_imagenet64_to_standard_mapping(imagenet64_path, standard_path)
    _standard_to_imagenet64 = {v: k for k, v in forward.items()}

    return _standard_to_imagenet64


def remap_imagenet64_labels_to_standard(
    labels: np.ndarray,
    imagenet64_path: Optional[Path] = None,
    standard_path: Optional[Path] = None
) -> np.ndarray:
    """
    Remap ImageNet64 labels to standard ImageNet indices.

    Args:
        labels: Array of ImageNet64 class indices (0-999 or 0-500 for subset)
        imagenet64_path: Optional path to imagenet64_class_labels.json
        standard_path: Optional path to imagenet_standard_class_index.json

    Returns:
        Array of standard ImageNet class indices (0-999)
    """
    mapping = load_imagenet64_to_standard_mapping(imagenet64_path, standard_path)

    # Vectorized remapping
    remapped = np.zeros_like(labels)
    for idx64, idx_std in mapping.items():
        remapped[labels == idx64] = idx_std

    return remapped


def remap_imagenet64_label(
    label: int,
    imagenet64_path: Optional[Path] = None,
    standard_path: Optional[Path] = None
) -> int:
    """
    Remap a single ImageNet64 label to standard ImageNet index.

    Args:
        label: ImageNet64 class index (0-999)

    Returns:
        Standard ImageNet class index (0-999)
    """
    mapping = load_imagenet64_to_standard_mapping(imagenet64_path, standard_path)
    return mapping.get(label, label)


def get_class_info(
    imagenet64_idx: int,
    imagenet64_path: Optional[Path] = None,
    standard_path: Optional[Path] = None
) -> dict:
    """
    Get class information for debugging.

    Returns dict with:
        - imagenet64_idx: Original ImageNet64 index
        - standard_idx: Corresponding standard ImageNet index
        - synset: WordNet synset ID
        - name: Human-readable class name
    """
    data_dir = get_data_dir()

    if imagenet64_path is None:
        imagenet64_path = data_dir / "imagenet64_class_labels.json"

    with open(imagenet64_path, 'r') as f:
        imagenet64_labels = json.load(f)

    mapping = load_imagenet64_to_standard_mapping(imagenet64_path, standard_path)

    idx_str = str(imagenet64_idx)
    if idx_str in imagenet64_labels:
        synset, name = imagenet64_labels[idx_str]
        return {
            'imagenet64_idx': imagenet64_idx,
            'standard_idx': mapping.get(imagenet64_idx, -1),
            'synset': synset,
            'name': name
        }
    return None


if __name__ == "__main__":
    # Test the mapping
    print("Testing ImageNet class mapping...")

    mapping = load_imagenet64_to_standard_mapping()

    # Test a few known mappings
    test_cases = [
        (0, 278, "kit_fox"),      # ImageNet64[0] -> Standard[278]
        (448, 0, "tench"),        # ImageNet64[448] -> Standard[0]
        (449, 1, "goldfish"),     # ImageNet64[449] -> Standard[1]
        (1, 212, "English_setter"),  # ImageNet64[1] -> Standard[212]
    ]

    print("\nVerifying known mappings:")
    all_passed = True
    for idx64, expected_std, name in test_cases:
        actual_std = mapping.get(idx64, -1)
        status = "✓" if actual_std == expected_std else "✗"
        if actual_std != expected_std:
            all_passed = False
        print(f"  {status} ImageNet64[{idx64}] ({name}) -> Standard[{actual_std}] (expected {expected_std})")

    # Test array remapping
    print("\nTesting array remapping:")
    test_labels = np.array([0, 1, 448, 449])
    remapped = remap_imagenet64_labels_to_standard(test_labels)
    print(f"  Input:  {test_labels}")
    print(f"  Output: {remapped}")
    print(f"  Expected: [278, 212, 0, 1]")

    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
