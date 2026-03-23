"""
EfficientNet-B0 Image-Only Galaxy Morphology Classifier.

Architecture:
    EfficientNet-B0 (pretrained on ImageNet, backbone optionally frozen)
    → AdaptiveAvgPool → Dropout(0.2) → 1280-d embedding
    → Linear(1280, num_classes) classification head

Compared to the ResNet-18 baseline:
    - Fewer parameters (~5.3M vs 11.2M) but typically higher accuracy
    - Built-in Squeeze-and-Excitation attention modules
    - Built-in Dropout in the classifier head (reduces overfitting)

Usage:
    from efficientnet_model import EfficientNetB0Classifier

    model = EfficientNetB0Classifier(num_classes=4)
    model.to(device)
"""

import torch
import torch.nn as nn
from torchvision import models


class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 with optional frozen backbone for galaxy morphology classification.

    The pretrained EfficientNet-B0 backbone learns rich visual representations.
    When freeze_backbone=False (--fine-tune), the entire network adapts to galaxies.
    """

    def __init__(self, num_classes=4, freeze_backbone=False):
        super().__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Optionally freeze all backbone layers apart from the classifier
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        # Replace the final classifier head
        # EfficientNet-B0's classifier is: Sequential(Dropout(0.2), Linear(1280, 1000))
        # We keep the Dropout but replace the Linear to match our num_classes
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

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
        Extract the 1280-d embedding from the layer BEFORE the classifier.
        Useful for embedding analysis and visualization.

        Args:
            x: Image tensor [B, 3, 224, 224]

        Returns:
            embedding: [B, 1280]
        """
        # EfficientNet: features → avgpool → flatten → classifier
        # We run only up to and including avgpool
        features = self.backbone.features(x)                   # [B, 1280, 7, 7]
        pooled = self.backbone.avgpool(features)               # [B, 1280, 1, 1]
        embedding = torch.flatten(pooled, 1)                   # [B, 1280]
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
    print("  EFFICIENTNET-B0 MODEL SELF-TEST")
    print("=" * 50)

    # Test with frozen backbone (default extraction mode)
    print("\n[1] Frozen backbone (feature extraction):")
    model_frozen = EfficientNetB0Classifier(num_classes=4, freeze_backbone=True)
    model_frozen.count_params()

    # Test with unfrozen backbone (fine-tune mode)
    print("\n[2] Unfrozen backbone (fine-tune mode):")
    model_finetune = EfficientNetB0Classifier(num_classes=4, freeze_backbone=False)
    model_finetune.count_params()

    print("\n[3] Forward pass test (CPU, batch=2)...")
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model_finetune(dummy_input)
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {logits.shape}")
    assert logits.shape == (2, 4), f"Expected (2, 4), got {logits.shape}"

    print("\n[4] Embedding extraction test...")
    emb = model_finetune.get_embedding(dummy_input)
    print(f"  Embedding shape: {emb.shape}")
    assert emb.shape == (2, 1280), f"Expected (2, 1280), got {emb.shape}"

    print("\n✅ All checks passed!")
