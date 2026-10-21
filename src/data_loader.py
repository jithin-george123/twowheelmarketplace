"""
Data loading and augmentation pipeline for Car Spare Parts.
Implements PyTorch Dataset, stratified data split loading,
data augmentations, normalization, and class-weight computation for loss balancing.
"""
import json
from pathlib import Path
from typing import Tuple, Dict, Any
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    from .config import (
        IMAGE_SIZE, BATCH_SIZE, NORM_MEAN, NORM_STD,
        AUGMENTATION_CONFIG, SPLIT_FILE
    )
except ImportError:
    from config import (
        IMAGE_SIZE, BATCH_SIZE, NORM_MEAN, NORM_STD,
        AUGMENTATION_CONFIG, SPLIT_FILE
    )


class SparePartsDataset(Dataset):
    """PyTorch Dataset for Car Spare Parts."""

    def __init__(self, records: list, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.records[idx]
        img_path = item["filepath"]
        label = item["label"]

        # Load image safely in RGB
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, img_path  # type: ignore


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Build train and validation/test transformation pipelines.
    
    Training pipeline uses mild augmentations preserving part identity.
    Validation/Test uses pure deterministic resize and normalization.
    """
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=AUGMENTATION_CONFIG["hflip_prob"]),
        transforms.RandomRotation(degrees=AUGMENTATION_CONFIG["rotation_degrees"]),
        transforms.ColorJitter(
            brightness=AUGMENTATION_CONFIG["brightness_jitter"],
            contrast=AUGMENTATION_CONFIG["contrast_jitter"]
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    return train_transform, val_transform


def compute_class_weights(train_records: list, num_classes: int) -> torch.Tensor:
    """Compute balanced class weights using inverse class frequency.
    
    w_j = N / (num_classes * N_j)
    Helps prevent majority-class bias during CrossEntropyLoss optimization.
    """
    labels = [r["label"] for r in train_records]
    counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)

    # Avoid zero-division if a class has 0 samples
    weights = total_samples / (num_classes * np.maximum(counts, 1).astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def get_data_loaders(
    split_file: Path = SPLIT_FILE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any], torch.Tensor]:
    """Instantiate DataLoaders for Train, Validation, and Test sets."""
    if not Path(split_file).exists():
        raise FileNotFoundError(f"Split file not found at {split_file}. Run dataset_inspector first.")

    with open(split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_transform, val_transform = get_transforms()

    train_dataset = SparePartsDataset(splits["train"], transform=train_transform)
    val_dataset = SparePartsDataset(splits["val"], transform=val_transform)
    test_dataset = SparePartsDataset(splits["test"], transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    num_classes = splits["metadata"]["num_classes"]
    class_weights = compute_class_weights(splits["train"], num_classes)

    return train_loader, val_loader, test_loader, splits["metadata"], class_weights
