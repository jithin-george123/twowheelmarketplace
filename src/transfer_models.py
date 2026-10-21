import torch.nn as nn
import timm

def create_transfer_model(model_name: str = "xception", num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Instantiate the Xception architecture for car spare parts recognition.
    """
    return timm.create_model("xception", pretrained=pretrained, num_classes=num_classes)
