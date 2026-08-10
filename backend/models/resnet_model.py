import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


def build_resnet50_model(num_classes: int = 4, freeze_backbone: bool = True) -> nn.Module:
    """
    Builds a ResNet50 model using ImageNet-pretrained weights from torchvision.
    
    Args:
        num_classes (int): Number of target classification classes (default: 4).
        freeze_backbone (bool): If True, freezes all parameters of the pretrained backbone.
        
    Returns:
        nn.Module: PyTorch ResNet50 model with updated classification head.
    """
    # Use modern torchvision weights API
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # Freeze pretrained backbone weights if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def get_parameter_counts(model: nn.Module) -> Tuple[int, int]:
    """
    Calculates total and trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model.
        
    Returns:
        Tuple[int, int]: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    model = build_resnet50_model(num_classes=4, freeze_backbone=True)
    total, trainable = get_parameter_counts(model)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Backbone frozen: {total > trainable}")
