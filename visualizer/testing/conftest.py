"""
Pytest configuration and shared fixtures for visualizer tests.
"""

import pytest
import sys
from pathlib import Path

# Add visualizer to path for imports
visualizer_dir = Path(__file__).parent.parent
if str(visualizer_dir) not in sys.path:
    sys.path.insert(0, str(visualizer_dir))

# Add project root for third_party imports
project_root = visualizer_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    dirs = {
        'root': tmp_path,
        'images': tmp_path / 'images' / 'imagenet_real',
        'activations': tmp_path / 'activations' / 'imagenet_real',
        'metadata': tmp_path / 'metadata' / 'imagenet_real',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture
def sample_activation():
    """Create sample activation tensor."""
    import torch
    return torch.randn(1, 768, 4, 4)


@pytest.fixture
def sample_image():
    """Create sample uint8 image tensor."""
    import torch
    return torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
