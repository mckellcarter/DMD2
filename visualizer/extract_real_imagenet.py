"""
Extract activations from real ImageNet images using DMD2 model.
Processes ImageNet validation/train set and saves activations in batch format.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from accelerate.utils import set_seed

# ImageNet imports
from third_party.edm.training.networks import EDMPrecond
from main.edm.edm_network import get_imagenet_edm_config

# Local imports
from extract_activations import ActivationExtractor
from device_utils import get_device, get_device_info, move_to_device


def get_imagenet_config():
    """Get ImageNet EDM config."""
    base_config = {
        "img_resolution": 64,
        "img_channels": 3,
        "label_dim": 1000,
        "use_fp16": False,
        "sigma_min": 0,
        "sigma_max": float("inf"),
        "sigma_data": 0.5,
        "model_type": "DhariwalUNet"
    }
    base_config.update(get_imagenet_edm_config())
    return base_config


def load_imagenet_model(checkpoint_path: str, device: str = "cuda"):
    """Load pretrained ImageNet generator."""
    base_config = get_imagenet_config()
    generator = EDMPrecond(**base_config)
    del generator.model.map_augment
    generator.model.map_augment = None

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    generator.load_state_dict(state_dict, strict=True)
    generator = move_to_device(generator, device)
    generator.eval()

    return generator


def preprocess_imagenet_image(image_path: Path, target_size: int = 64) -> torch.Tensor:
    """
    Load and preprocess ImageNet image to match DMD2 input format.

    Args:
        image_path: Path to image file
        target_size: Target resolution (default 64 for ImageNet-64)

    Returns:
        Preprocessed tensor (1, 3, H, W) in range [-1, 1]
    """
    img = Image.open(image_path).convert('RGB')

    # Resize to target resolution
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # Convert to tensor and normalize to [-1, 1]
    img_array = np.array(img).astype(np.float32)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (3, H, W)
    img_tensor = (img_tensor / 127.5) - 1.0  # [0, 255] -> [-1, 1]
    img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)

    return img_tensor


def parse_imagenet_path(image_path: Path) -> Tuple[str, Optional[int]]:
    """
    Parse ImageNet image path to extract synset ID.

    Assumes structure like:
        - imagenet/val/n01440764/ILSVRC2012_val_00000001.JPEG
        - imagenet/train/n01440764/n01440764_1.JPEG

    Args:
        image_path: Path to ImageNet image

    Returns:
        (synset_id, class_id_if_known)
    """
    # Parent directory should be synset ID (e.g., n01440764)
    synset_id = image_path.parent.name

    # Try to match synset to class ID
    # Will be resolved using class_labels_map in main function
    return synset_id, None


def extract_real_imagenet_activations(
    checkpoint_path: str,
    imagenet_dir: Path,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    layers: List[str],
    conditioning_sigma: float = 0.0,
    split: str = "val",
    seed: int = 10,
    device: str = None
):
    """
    Extract activations from real ImageNet images.

    Args:
        checkpoint_path: Path to DMD2 model checkpoint
        imagenet_dir: Root directory of ImageNet dataset
        output_dir: Output directory for activations
        num_samples: Total samples to process
        batch_size: Batch size for processing
        layers: List of layer names to extract
        conditioning_sigma: Sigma for forward pass (default 0.0 for clean reconstruction)
        split: Dataset split ("val" or "train")
        seed: Random seed for shuffling
        device: Device to use (auto-detect if None)
    """
    # Auto-detect device if not specified
    if device is None:
        device = get_device()

    device_info = get_device_info(device)
    print(f"\nDevice: {device_info['device']} ({device_info['device_name']})")
    if device_info['memory_allocated'] != 'N/A':
        print(f"GPU Memory: {device_info['memory_allocated']:.2f} GB allocated")

    print(f"\nLoading ImageNet model from {checkpoint_path}")
    generator = load_imagenet_model(checkpoint_path, device)

    # Setup extractor
    extractor = ActivationExtractor(model_type="imagenet")
    extractor.register_hooks(generator, layers)

    # Load ImageNet class labels
    class_labels_path = Path(__file__).parent / "data" / "imagenet_class_labels.json"
    with open(class_labels_path, 'r', encoding='utf-8') as f:
        class_labels_map = json.load(f)

    # Create reverse mapping: synset_id -> (class_id, class_name)
    synset_to_class = {}
    for class_id_str, (synset_id, class_name) in class_labels_map.items():
        synset_to_class[synset_id] = (int(class_id_str), class_name)

    # Create output directories
    image_dir = output_dir / "images" / "imagenet_real"
    reconstructed_dir = output_dir / "images" / "imagenet_real_reconstructed"
    activation_dir = output_dir / "activations" / "imagenet_real"
    metadata_dir = output_dir / "metadata" / "imagenet_real"
    image_dir.mkdir(parents=True, exist_ok=True)
    reconstructed_dir.mkdir(parents=True, exist_ok=True)
    activation_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Find ImageNet images
    imagenet_split_dir = imagenet_dir / split
    if not imagenet_split_dir.exists():
        raise FileNotFoundError(
            f"ImageNet split directory not found: {imagenet_split_dir}\n"
            f"Expected structure: {imagenet_dir}/{split}/n01440764/*.JPEG"
        )

    # Collect all image paths
    image_extensions = ['.JPEG', '.jpg', '.png']
    all_image_paths = []
    for ext in image_extensions:
        all_image_paths.extend(list(imagenet_split_dir.rglob(f'*{ext}')))

    print(f"\nFound {len(all_image_paths)} images in {imagenet_split_dir}")

    if len(all_image_paths) == 0:
        raise ValueError(f"No images found in {imagenet_split_dir}")

    # Shuffle and limit
    set_seed(seed)
    np.random.shuffle(all_image_paths)
    all_image_paths = all_image_paths[:num_samples]

    print(f"Processing {len(all_image_paths)} images")

    # Process in batches
    sample_idx = 0
    all_metadata = []
    num_batches = (len(all_image_paths) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_image_paths))
        batch_paths = all_image_paths[start_idx:end_idx]
        current_batch_size = len(batch_paths)

        # Load and preprocess batch
        batch_images = []
        batch_labels = []
        batch_synsets = []
        batch_class_names = []
        batch_original_paths = []

        for img_path in batch_paths:
            # Preprocess image
            img_tensor = preprocess_imagenet_image(img_path)
            batch_images.append(img_tensor)

            # Parse synset from path
            synset_id, _ = parse_imagenet_path(img_path)

            # Lookup class info
            if synset_id in synset_to_class:
                class_id, class_name = synset_to_class[synset_id]
            else:
                print(f"Warning: Unknown synset {synset_id} for {img_path}")
                class_id = -1
                class_name = "unknown"

            batch_labels.append(class_id)
            batch_synsets.append(synset_id)
            batch_class_names.append(class_name)
            batch_original_paths.append(str(img_path))

        # Stack batch
        batch_tensor = torch.cat(batch_images, dim=0).to(device)  # (B, 3, 64, 64)
        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)
        one_hot_labels = torch.eye(1000, device=device)[batch_labels_tensor]

        # Extract activations by running forward pass
        extractor.clear_activations()
        with torch.no_grad():
            sigma = torch.ones(current_batch_size, device=device) * conditioning_sigma
            # Run through generator to extract activations AND get reconstructed output
            reconstructed_images = generator(
                batch_tensor * conditioning_sigma,
                sigma,
                one_hot_labels
            )

        # Convert reconstructed images to uint8
        reconstructed_images_uint8 = (
            ((reconstructed_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        )
        reconstructed_images_uint8 = (
            reconstructed_images_uint8.permute(0, 2, 3, 1).cpu().numpy()
        )

        # Get activations
        activations = extractor.get_activations()

        # Save batch activations
        batch_id = f"batch_{batch_idx:06d}"
        batch_act_path = activation_dir / batch_id

        # Save activations (NPZ)
        activation_dict = {}
        for name, activation in activations.items():
            if len(activation.shape) == 4:
                batch_dim = activation.shape[0]
                activation_dict[name] = activation.reshape(batch_dim, -1).cpu().numpy()
            else:
                activation_dict[name] = activation.cpu().numpy()

        np.savez_compressed(
            str(batch_act_path.with_suffix('.npz')),
            **activation_dict
        )

        # Save batch metadata (JSON) with ImageNet identifiers
        batch_samples_meta = []
        for i in range(current_batch_size):
            batch_samples_meta.append({
                "batch_index": i,
                "class_id": int(batch_labels[i]),
                "synset_id": batch_synsets[i],
                "class_name": batch_class_names[i],
                "original_path": batch_original_paths[i]
            })

        batch_metadata = {
            "batch_size": current_batch_size,
            "layers": layers,
            "samples": batch_samples_meta
        }

        with open(batch_act_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(batch_metadata, f, indent=2)

        # Copy images to output directory and track metadata
        for i in range(current_batch_size):
            sample_id = f"sample_{sample_idx:06d}"

            # Copy original image
            img_path = image_dir / f"{sample_id}.JPEG"
            Image.open(batch_paths[i]).save(img_path)

            # Save reconstructed image
            reconstructed_path = reconstructed_dir / f"{sample_id}.png"
            Image.fromarray(reconstructed_images_uint8[i]).save(reconstructed_path)

            # Track metadata
            all_metadata.append({
                "sample_id": sample_id,
                "class_label": int(batch_labels[i]),
                "synset_id": batch_synsets[i],
                "class_name": batch_class_names[i],
                "image_path": str(img_path.relative_to(output_dir)),
                "reconstructed_path": str(reconstructed_path.relative_to(output_dir)),
                "activation_path": str(batch_act_path.relative_to(output_dir)),
                "batch_index": i,
                "original_path": batch_original_paths[i],
                "source": "imagenet_real",
                "conditioning_sigma": conditioning_sigma
            })

            sample_idx += 1

    # Save global metadata
    metadata_path = metadata_dir / "dataset_info.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model_type": "imagenet_real",
            "num_samples": sample_idx,
            "layers": layers,
            "conditioning_sigma": conditioning_sigma,
            "seed": seed,
            "split": split,
            "class_labels": class_labels_map,
            "samples": all_metadata
        }, f, indent=2)

    print(f"\nProcessed {sample_idx} real ImageNet samples")
    print(f"Original Images: {image_dir}")
    print(f"Reconstructed Images: {reconstructed_dir}")
    print(f"Activations: {activation_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Conditioning Sigma: {conditioning_sigma}")

    extractor.remove_hooks()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Extract activations from real ImageNet images"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to DMD2 model checkpoint"
    )
    parser.add_argument(
        "--imagenet_dir",
        type=str,
        required=True,
        help="Root directory of ImageNet dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for activations"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to process"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="encoder_bottleneck,midblock",
        help="Comma-separated list of layers to extract"
    )
    parser.add_argument(
        "--conditioning_sigma",
        type=float,
        default=0.0,
        help="Conditioning sigma for forward pass (0.0 = clean reconstruction, 80.0 = generation)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "train"],
        help="ImageNet split to process"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (auto-detect if not specified)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    imagenet_dir = Path(args.imagenet_dir)
    layers = args.layers.split(",")

    extract_real_imagenet_activations(
        checkpoint_path=args.checkpoint_path,
        imagenet_dir=imagenet_dir,
        output_dir=output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        layers=layers,
        conditioning_sigma=args.conditioning_sigma,
        split=args.split,
        seed=args.seed,
        device=args.device
    )


if __name__ == "__main__":
    main()
