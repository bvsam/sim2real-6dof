"""
ResNet-50 + FPN Backbone for NOCS R-CNN.
Uses torchvision pretrained weights.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import FeaturePyramidNetwork


class ResNetFPNBackbone(nn.Module):
    """
    ResNet-50 backbone with Feature Pyramid Network (FPN).

    Returns feature maps at multiple scales: [P2, P3, P4, P5]
    - P2: stride 4  (1/4 resolution)
    - P3: stride 8  (1/8 resolution)
    - P4: stride 16 (1/16 resolution)
    - P5: stride 32 (1/32 resolution)
    """

    def __init__(self, pretrained=True, freeze_bn=True):
        """
        Args:
            pretrained: Whether to use ImageNet pretrained weights
            freeze_bn: Whether to freeze batch norm layers
        """
        super().__init__()

        # Load pretrained ResNet-50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
            resnet = resnet50(weights=weights)
        else:
            resnet = resnet50(weights=None)

        # Extract ResNet layers (remove avgpool and fc)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # C2 (stride 4, 256 channels)
        self.layer2 = resnet.layer2  # C3 (stride 8, 512 channels)
        self.layer3 = resnet.layer3  # C4 (stride 16, 1024 channels)
        self.layer4 = resnet.layer4  # C5 (stride 32, 2048 channels)

        # Feature Pyramid Network
        in_channels_list = [256, 512, 1024, 2048]  # C2, C3, C4, C5
        out_channels = 256

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
        )

        self.out_channels = out_channels

        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def freeze_stages(self, stage):
        """
        Freeze ResNet stages for progressive training.

        Args:
            stage: 0 = freeze nothing
                   1 = freeze conv1, bn1
                   2 = freeze stage 1 + layer1 (C2)
                   3 = freeze stage 2 + layer2 (C3)
                   4 = freeze stage 3 + layer3 (C4)
                   5 = freeze all
        """
        if stage >= 1:
            # Freeze conv1, bn1
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False

        if stage >= 2:
            # Freeze layer1 (C2)
            for param in self.layer1.parameters():
                param.requires_grad = False

        if stage >= 3:
            # Freeze layer2 (C3)
            for param in self.layer2.parameters():
                param.requires_grad = False

        if stage >= 4:
            # Freeze layer3 (C4)
            for param in self.layer3.parameters():
                param.requires_grad = False

        if stage >= 5:
            # Freeze layer4 (C5)
            for param in self.layer4.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            OrderedDict of FPN features:
                - 'p2': [B, 256, H/4, W/4]
                - 'p3': [B, 256, H/8, W/8]
                - 'p4': [B, 256, H/16, W/16]
                - 'p5': [B, 256, H/32, W/32]
        """
        # ResNet stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet stages
        c2 = self.layer1(x)  # stride 4
        c3 = self.layer2(c2)  # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32

        # FPN
        features = OrderedDict(
            [
                ("c2", c2),
                ("c3", c3),
                ("c4", c4),
                ("c5", c5),
            ]
        )

        fpn_features = self.fpn(features)

        # Rename to P2, P3, P4, P5
        output = OrderedDict(
            [
                ("p2", fpn_features["c2"]),
                ("p3", fpn_features["c3"]),
                ("p4", fpn_features["c4"]),
                ("p5", fpn_features["c5"]),
            ]
        )

        return output


if __name__ == "__main__":
    # Test the backbone
    print("Testing ResNet-50 + FPN Backbone...")

    model = ResNetFPNBackbone(pretrained=True)

    # Test input
    x = torch.randn(2, 3, 480, 640)  # Your image size

    # Forward pass
    features = model(x)

    print("\nOutput feature maps:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")

    # Test freezing
    print("\nTesting stage freezing...")
    model.freeze_stages(3)  # Freeze up to C3

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Trainable params: {trainable_params:,} / {total_params:,}")
    print("✓ Backbone test passed!")
