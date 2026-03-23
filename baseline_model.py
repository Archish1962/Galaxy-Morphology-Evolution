"""
Baseline Image-Only Galaxy Morphology Classifier.

Architecture:
    ResNet18 (pretrained on ImageNet, backbone frozen)
    → AdaptiveAvgPool → 512-d embedding
    → Linear(512, 4) classification head

Only the final classification head is trained.
The frozen backbone acts as a feature extractor.

Usage:
    from baseline_model import BaselineResNet18

    model = BaselineResNet18(num_classes=4)
    model.to(device)
"""

import torch
import torch.nn as nn
from torchvision import models


class BaselineResNet18(nn.Module):
    """
    ResNet18 with frozen backbone for galaxy morphology classification.

    The pretrained backbone extracts visual features. Only the final
    classification layer is trained, making this fast to converge and
    serving as a performance baseline for the multimodal model.
    """

    def __init__(self, num_classes=4, freeze_backbone=True):
        super().__init__()

        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze all backbone layers (conv1 through layer4 + bn/avgpool)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final FC layer (originally 512 → 1000 for ImageNet)
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Linear(in_features, num_classes)
        # This new layer has requires_grad=True by default

    def forward(self, x):
        """
        Args:
            x: Image tensor [B, 3, 224, 224]

        Returns:
            logits: [B, num_classes]
        """
        return self.backbone(x)

    def get_embedding(self, x):
        """
        Extract the 512-d embedding from the layer BEFORE the classifier.
        Useful for Phase 3 embedding analysis.

        Args:
            x: Image tensor [B, 3, 224, 224]

        Returns:
            embedding: [B, 512]
        """
        # Forward through all layers except the final FC
        modules = list(self.backbone.children())[:-1]  # everything except fc
        feature_extractor = nn.Sequential(*modules)
        embedding = feature_extractor(x)
        embedding = embedding.squeeze(-1).squeeze(-1)  # [B, 512, 1, 1] → [B, 512]
        return embedding

    def get_trainable_params(self):
        """Return only trainable parameters (for the optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_params(self):
        """Print trainable vs total parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"  Total params:     {total:,}")
        print(f"  Trainable params: {trainable:,}  ({trainable/total:.1%})")
        print(f"  Frozen params:    {frozen:,}  ({frozen/total:.1%})")
        return trainable, total


# === Quick self-test (no GPU required) ===
if __name__ == "__main__":
    print("=" * 50)
    print("  BASELINE MODEL SELF-TEST")
    print("=" * 50)

    model = BaselineResNet18(num_classes=4)

    print("\n[1] Parameter counts:")
    model.count_params()

    print("\n[2] Forward pass test (CPU, batch=2)...")
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Logits:       {logits.detach()}")

    print("\n[3] Embedding extraction test...")
    emb = model.get_embedding(dummy_input)
    print(f"  Embedding shape: {emb.shape}")

    print("\n✅ All checks passed!")
