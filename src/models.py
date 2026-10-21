"""
Xception Model Architecture for Car Spare Parts Recognition.
"""
import torch
import torch.nn as nn
import timm
from typing import Dict, Any


class XceptionSparePartsModel(nn.Module):
    """
    Xception deep convolutional neural network for 10-class car spare parts recognition.
    """
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model("xception", pretrained=pretrained, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Calculate total trainable parameters and model size in megabytes."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (total_params * 4) / (1024 * 1024)  # float32 = 4 bytes
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(size_mb, 2)
    }
