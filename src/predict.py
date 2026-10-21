"""
Standalone Inference and Prediction System for Car Spare Parts Recognition.
Accepts an image path, executes the trained Xception inference pipeline,
returns the Top-3 predictions with confidence scores, and connects to marketplace metadata.
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    from .config import (
        DEVICE, IMAGE_SIZE, NORM_MEAN, NORM_STD,
        XCEPTION_MODEL_PATH, CLASS_MAPPING_FILE
    )
    from .transfer_models import create_transfer_model
    from .marketplace import MarketplaceCatalog
except ImportError:
    from config import (
        DEVICE, IMAGE_SIZE, NORM_MEAN, NORM_STD,
        XCEPTION_MODEL_PATH, CLASS_MAPPING_FILE
    )
    from transfer_models import create_transfer_model
    from marketplace import MarketplaceCatalog


class SparePartPredictor:
    """Predictor service for inference on car spare part images using Xception."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        class_mapping_path: Optional[Union[str, Path]] = None
    ):
        self.model_path = Path(model_path) if model_path else XCEPTION_MODEL_PATH
        self.class_mapping_path = Path(class_mapping_path) if class_mapping_path else CLASS_MAPPING_FILE
        self.device = DEVICE
        self.model_name = "xception"

        # Load class mappings
        if not self.class_mapping_path.exists():
            raise FileNotFoundError(f"Class mapping not found at {self.class_mapping_path}.")
        with open(self.class_mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
            self.idx_to_class = {int(k): v for k, v in mapping_data["idx_to_class"].items()}
            self.class_to_idx = mapping_data["class_to_idx"]

        self.num_classes = len(self.idx_to_class)
        self.marketplace = MarketplaceCatalog()

        # Load Xception model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Xception checkpoint not found at {self.model_path}.")

        self.model = create_transfer_model("xception", num_classes=self.num_classes, pretrained=False)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Inference image transform
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])

    def predict(
        self,
        image_path: Union[str, Path],
        include_marketplace: bool = True
    ) -> Dict[str, Any]:
        """Perform inference on single spare part image and return top-3 predictions."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Open and preprocess image
        try:
            with Image.open(image_path) as img:
                image_rgb = img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not open or decode image: {e}")

        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Rank predictions
        ranked_indices = probabilities.argsort()[::-1]
        top_indices = ranked_indices[:min(3, len(ranked_indices))]

        top_3 = []
        for rank, idx in enumerate(top_indices, start=1):
            c_name = self.idx_to_class[int(idx)]
            top_3.append({
                "rank": rank,
                "class_name": c_name,
                "display_name": c_name.replace("_", " ").title(),
                "confidence": float(probabilities[idx]),
                "confidence_percent": f"{probabilities[idx] * 100:.1f}%"
            })

        predicted_class = top_3[0]["class_name"]
        predicted_confidence = top_3[0]["confidence"]

        result = {
            "image_path": str(image_path),
            "predicted_class": predicted_class,
            "display_name": top_3[0]["display_name"],
            "confidence": predicted_confidence,
            "confidence_percent": top_3[0]["confidence_percent"],
            "top_3": top_3
        }

        if include_marketplace:
            result["marketplace_info"] = self.marketplace.get_part_info(predicted_class)

        return result


_default_predictor = None

def predict_spare_part(
    image_path: str,
    model_name: Optional[str] = None,
    model_path: Optional[Union[str, Path]] = None,
    print_output: bool = True,
    show_marketplace: bool = True
) -> Dict[str, Any]:
    """Top-level function to predict spare part class from image path using Xception."""
    global _default_predictor
    if _default_predictor is None or model_path is not None:
        predictor = SparePartPredictor(model_path=model_path)
        if model_path is None:
            _default_predictor = predictor
    else:
        predictor = _default_predictor

    result = predictor.predict(image_path, include_marketplace=show_marketplace)

    if print_output:
        info = result.get("marketplace_info") or predictor.marketplace.get_part_info(result["predicted_class"])
        product_name = info.get("display_name", result["display_name"])
        category = info.get("category", "General Automotive Parts")

        print("\n" + "=" * 60)
        print(f"  Product Name:  {product_name}")
        print(f"  Category:      {category}")
        print(f"  AI Detection:  {result['display_name']} ({result['confidence_percent']} confidence)")
        print("=" * 60 + "\n")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classify car spare parts using Xception")
    parser.add_argument("image_path", type=str, help="Path to car spare part image")
    parser.add_argument("--weights", type=str, default=None, help="Custom checkpoint path (optional)")
    args = parser.parse_args()

    predict_spare_part(
        args.image_path,
        model_path=args.weights,
        print_output=True,
        show_marketplace=True
    )
