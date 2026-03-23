"""
Multimodal Fusion Model for Galaxy Morphology Classification.

Supports two image backbones:
    - resnet18:       512-d image embedding  → Fusion input = 512 + 128 = 640
    - efficientnet_b0: 1280-d image embedding → Fusion input = 1280 + 128 = 1408

Architecture:
    Image Branch:  Backbone (pretrained) → N-d image embedding
    Meta Branch:   MLP [6 → 64 → 128] with ReLU + BatchNorm
    Fusion:        Concat(N, 128) → Linear(N+128, 256) → ReLU → Linear(256, 4)

The 256-d layer before the final classifier is the FUSED EMBEDDING
used for centroid drift analysis in Phase 3.

Usage:
    from multimodal_model import MultimodalFusionNet

    model = MultimodalFusionNet(num_classes=4, backbone="resnet18")
    model = MultimodalFusionNet(num_classes=4, backbone="efficientnet_b0")
    logits = model(images, metadata)
    embedding = model.get_embedding(images, metadata)
"""

import torch
import torch.nn as nn
from torchvision import models


class MetadataMLP(nn.Module):
    """
    Small MLP to process galaxy metadata features.

    Input:  [B, num_features]  (redshift, colors, structural features)
    Output: [B, 128]           (metadata embedding)
    """

    def __init__(self, num_features=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.network(x)


def _build_image_branch(backbone_name: str, freeze_backbone: bool):
    """
    Build the image feature extractor and return (extractor_fn, embed_dim).

    Returns a callable that takes [B, 3, 224, 224] and returns [B, embed_dim].
    """
    if backbone_name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if freeze_backbone:
            for param in net.parameters():
                param.requires_grad = False
        # Remove the classifier; keep everything up to avgpool
        backbone = nn.Sequential(*list(net.children())[:-1])
        # Output: [B, 512, 1, 1] — squeezed to [B, 512] in forward()
        embed_dim = 512

    elif backbone_name == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        if freeze_backbone:
            for name, param in net.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
        # Keep just the feature extractor + avgpool
        backbone = nn.Sequential(net.features, net.avgpool)
        # Output: [B, 1280, 1, 1] — squeezed to [B, 1280] in forward()
        embed_dim = 1280

    else:
        raise ValueError(f"Unknown backbone: '{backbone_name}'. Choose 'resnet18' or 'efficientnet_b0'.")

    return backbone, embed_dim


class MultimodalFusionNet(nn.Module):
    """
    Multimodal galaxy classifier combining image and metadata features.

    The image branch uses a pretrained backbone (ResNet-18 or EfficientNet-B0).
    The metadata branch uses a small MLP.
    Both embeddings are concatenated and passed through a fusion classifier.
    """

    def __init__(self, num_classes=4, num_meta_features=6,
                 backbone="resnet18", freeze_backbone=True):
        super().__init__()

        self.backbone_name = backbone

        # --- Image Branch ---
        self.image_backbone, self.img_embed_dim = _build_image_branch(
            backbone, freeze_backbone
        )

        # --- Metadata Branch: MLP ---
        self.meta_mlp = MetadataMLP(num_features=num_meta_features)
        # Output: [B, 128]

        # --- Fusion Classifier ---
        fusion_input_dim = self.img_embed_dim + 128
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes),
        )

    def _extract_image_embedding(self, images):
        """Run backbone and flatten spatial dims → [B, embed_dim]."""
        out = self.image_backbone(images)    # [B, C, 1, 1]
        return out.squeeze(-1).squeeze(-1)   # [B, C]

    def forward(self, images, metadata):
        """
        Args:
            images:   [B, 3, 224, 224]
            metadata: [B, num_meta_features]

        Returns:
            logits:   [B, num_classes]
        """
        img_emb = self._extract_image_embedding(images)   # [B, embed_dim]
        meta_emb = self.meta_mlp(metadata)                # [B, 128]
        fused = torch.cat([img_emb, meta_emb], dim=1)     # [B, embed_dim+128]
        return self.fusion_head(fused)

    def get_embedding(self, images, metadata):
        """
        Extract the 256-d fused embedding (before the final classifier).
        Used for Phase 3 centroid drift analysis.

        Returns:
            embedding: [B, 256]
        """
        img_emb = self._extract_image_embedding(images)
        meta_emb = self.meta_mlp(metadata)
        fused = torch.cat([img_emb, meta_emb], dim=1)

        # fusion_head: Linear → BN → ReLU → Dropout → Linear(final)
        # We extract after ReLU (index 2), skipping the final Linear
        embedding = self.fusion_head[0](fused)   # Linear → [B, 256]
        embedding = self.fusion_head[1](embedding)  # BN
        embedding = self.fusion_head[2](embedding)  # ReLU
        return embedding  # [B, 256]

    def get_trainable_params(self):
        """Return only trainable parameters (for the optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_params(self):
        """Print trainable vs total parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        print(f"  Backbone:         {self.backbone_name}")
        print(f"  Image embed dim:  {self.img_embed_dim}")
        print(f"  Total params:     {total:,}")
        print(f"  Trainable params: {trainable:,}  ({trainable/total:.1%})")
        print(f"  Frozen params:    {frozen:,}  ({frozen/total:.1%})")

        meta_params = sum(p.numel() for p in self.meta_mlp.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_head.parameters())
        print(f"  Metadata MLP:     {meta_params:,}")
        print(f"  Fusion head:      {fusion_params:,}")

        return trainable, total


# === Quick self-test ===
if __name__ == "__main__":
    print("=" * 55)
    print("  MULTIMODAL FUSION MODEL SELF-TEST")
    print("=" * 55)

    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_meta = torch.randn(2, 6)

    for backbone in ["resnet18", "efficientnet_b0"]:
        print(f"\n{'─'*55}")
        print(f"  Backbone: {backbone}")
        print(f"{'─'*55}")
        model = MultimodalFusionNet(num_classes=4, backbone=backbone, freeze_backbone=False)

        print("\n[1] Parameter counts:")
        model.count_params()

        print("\n[2] Forward pass test (CPU, batch=2)...")
        logits = model(dummy_images, dummy_meta)
        print(f"  Logits shape: {logits.shape}")
        assert logits.shape == (2, 4), f"Expected (2, 4), got {logits.shape}"

        print("\n[3] Embedding extraction test...")
        emb = model.get_embedding(dummy_images, dummy_meta)
        print(f"  Embedding shape: {emb.shape}")
        assert emb.shape == (2, 256), f"Expected (2, 256), got {emb.shape}"

        print("  ✅ Passed!")

    print(f"\n{'='*55}")
    print("  ALL CHECKS PASSED!")
    print(f"{'='*55}")
