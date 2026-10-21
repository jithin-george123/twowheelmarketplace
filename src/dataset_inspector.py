"""
Dataset Inspector and Split Generator for Car Spare Parts.
Automatically discovers the dataset, validates image integrity,
computes statistics, handles duplicates, and generates reproducible splits.
"""
import os
import json
import hashlib
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any
from PIL import Image

try:
    from .config import (
        DATASET_DIR, SPLIT_FILE, CLASS_MAPPING_FILE,
        TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED
    )
except ImportError:
    from config import (
        DATASET_DIR, SPLIT_FILE, CLASS_MAPPING_FILE,
        TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED
    )


class DatasetInspector:
    """Inspects car spare parts dataset, verifies integrity, and manages data splits."""

    def __init__(self, dataset_dir: Path = DATASET_DIR):
        self.dataset_dir = Path(dataset_dir)
        self.classes: List[str] = []
        self.class_counts: Dict[str, int] = {}
        self.corrupted_files: List[Tuple[str, str]] = []
        self.duplicates: List[Tuple[str, str]] = []
        self.valid_files: Dict[str, List[str]] = {}
        self.dimension_stats: Dict[str, Any] = {}

    def inspect(self) -> Dict[str, Any]:
        """Perform comprehensive inspection of dataset folder."""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found at: {self.dataset_dir}")

        # Detect class folders automatically
        subdirs = [d for d in self.dataset_dir.iterdir() if d.is_dir()]
        self.classes = sorted([d.name for d in subdirs])
        if not self.classes:
            raise ValueError(f"No class subdirectories found in {self.dataset_dir}")

        hashes = {}
        all_dims = []

        for cls_name in self.classes:
            cls_folder = self.dataset_dir / cls_name
            self.valid_files[cls_name] = []
            
            # Find all image files
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            image_paths = [p for p in cls_folder.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]

            for img_path in image_paths:
                # 1. Integrity check with PIL
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    with Image.open(img_path) as img:
                        all_dims.append(img.size)
                except Exception as e:
                    self.corrupted_files.append((str(img_path), str(e)))
                    continue

                # 2. Check duplicate hashes
                try:
                    with open(img_path, "rb") as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    if file_hash in hashes:
                        self.duplicates.append((str(img_path), hashes[file_hash]))
                    else:
                        hashes[file_hash] = str(img_path)
                except Exception:
                    pass

                self.valid_files[cls_name].append(str(img_path))

            self.class_counts[cls_name] = len(self.valid_files[cls_name])

        if all_dims:
            w_list, h_list = zip(*all_dims)
            self.dimension_stats = {
                "min": (min(w_list), min(h_list)),
                "max": (max(w_list), max(h_list)),
                "avg": (sum(w_list) // len(w_list), sum(h_list) // len(h_list))
            }

        summary = {
            "dataset_location": str(self.dataset_dir),
            "num_classes": len(self.classes),
            "classes": self.classes,
            "total_images": sum(self.class_counts.values()),
            "images_per_class": self.class_counts,
            "corrupted_images": len(self.corrupted_files),
            "duplicate_images": len(self.duplicates),
            "dimension_stats": self.dimension_stats
        }

        # Save class mapping
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        with open(CLASS_MAPPING_FILE, "w", encoding="utf-8") as f:
            json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2)

        return summary

    def create_or_load_splits(self, force_recreate: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Create stratified train/val/test split or load existing split file."""
        if SPLIT_FILE.exists() and not force_recreate:
            with open(SPLIT_FILE, "r", encoding="utf-8") as f:
                splits = json.load(f)
            return splits

        import random
        random.seed(RANDOM_SEED)

        train_records = []
        val_records = []
        test_records = []

        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls_name, paths in self.valid_files.items():
            # deterministic shuffle
            shuffled = list(paths)
            random.shuffle(shuffled)

            total = len(shuffled)
            n_train = int(total * TRAIN_SPLIT)
            n_val = int(total * VAL_SPLIT)
            
            cls_train = shuffled[:n_train]
            cls_val = shuffled[n_train:n_train + n_val]
            cls_test = shuffled[n_train + n_val:]

            cls_idx = class_to_idx[cls_name]

            for p in cls_train:
                train_records.append({"filepath": p, "class_name": cls_name, "label": cls_idx})
            for p in cls_val:
                val_records.append({"filepath": p, "class_name": cls_name, "label": cls_idx})
            for p in cls_test:
                test_records.append({"filepath": p, "class_name": cls_name, "label": cls_idx})

        # Shuffle splits
        random.shuffle(train_records)
        random.shuffle(val_records)
        random.shuffle(test_records)

        splits = {
            "train": train_records,
            "val": val_records,
            "test": test_records,
            "metadata": {
                "num_classes": len(self.classes),
                "classes": self.classes,
                "train_count": len(train_records),
                "val_count": len(val_records),
                "test_count": len(test_records),
                "total_count": len(train_records) + len(val_records) + len(test_records),
                "random_seed": RANDOM_SEED
            }
        }

        with open(SPLIT_FILE, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)

        return splits

    def print_summary(self, summary: Dict[str, Any], splits: Dict[str, Any]) -> None:
        """Display clean formatted summary of dataset and splits."""
        min_c = min(summary["images_per_class"].values())
        max_c = max(summary["images_per_class"].values())
        ratio = max_c / min_c if min_c > 0 else 0

        print("\n" + "=" * 65)
        print("  CAR SPARE PARTS DATASET SUMMARY")
        print("=" * 65)
        print(f"Dataset location:     {summary['dataset_location']}")
        print(f"Number of classes:    {summary['num_classes']}")
        print(f"Class names:          {summary['classes']}")
        print(f"Total valid images:   {summary['total_images']}")
        print("-" * 65)
        print("Images per class:")
        for cls, count in summary["images_per_class"].items():
            pct = (count / summary["total_images"]) * 100
            print(f"  - {cls:<16}: {count:>4} images ({pct:>5.1f}%)")
        print("-" * 65)
        print(f"Corrupted/unreadable: {summary['corrupted_images']}")
        print(f"Duplicate files:      {summary['duplicate_images']} intra-class duplicates")
        if summary.get("dimension_stats"):
            ds = summary["dimension_stats"]
            print(f"Image dimensions:     min {ds['min'][0]}x{ds['min'][1]} | max {ds['max'][0]}x{ds['max'][1]} | avg {ds['avg'][0]}x{ds['avg'][1]}")
        print("-" * 65)
        print("Data split distribution:")
        print(f"  - Training images:   {splits['metadata']['train_count']:>4} ({TRAIN_SPLIT * 100:.0f}%)")
        print(f"  - Validation images: {splits['metadata']['val_count']:>4} ({VAL_SPLIT * 100:.0f}%)")
        print(f"  - Test images:       {splits['metadata']['test_count']:>4} ({TEST_SPLIT * 100:.0f}%)")
        print(f"Detected imbalance:   Mild (Min: {min_c}, Max: {max_c}, Ratio: {ratio:.2f}:1)")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    inspector = DatasetInspector()
    summary = inspector.inspect()
    splits = inspector.create_or_load_splits(force_recreate=True)
    inspector.print_summary(summary, splits)
