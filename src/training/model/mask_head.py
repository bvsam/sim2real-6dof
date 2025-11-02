"""
Mask Prediction Head (standard Mask R-CNN component).
Predicts binary segmentation masks for each RoI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskHead(nn.Module):
    """
    Mask prediction head from Mask R-CNN.

    Architecture:
    - Input: 14x14x256 RoI features
    - 4x Conv2D (3x3, 256 channels) at 14x14
    - 1x ConvTranspose2D (2x2, stride=2) -> 28x28
    - 1x Conv2D (1x1) -> num_classes masks
    """

    def __init__(
        self,
        in_channels=256,
        num_conv_layers=4,
        conv_channels=256,
        num_classes=2,
        output_size=28,
    ):
        """
        Args:
            in_channels: Input channels from RoI features (256)
            num_conv_layers: Number of 3x3 conv layers (4 in Mask R-CNN)
            conv_channels: Channels in conv layers (256)
            num_classes: Number of object classes
            output_size: Output mask size (28x28)
        """
        super().__init__()

        self.num_classes = num_classes
        self.output_size = output_size

        # Convolutional layers
        conv_layers = []
        for i in range(num_conv_layers):
            in_ch = in_channels if i == 0 else conv_channels
            conv_layers.append(
                nn.Conv2d(in_ch, conv_channels, kernel_size=3, padding=1)
            )
            conv_layers.append(nn.ReLU(inplace=True))

        self.conv_layers = nn.Sequential(*conv_layers)

        # Upsampling layer
        self.deconv = nn.ConvTranspose2d(
            conv_channels, conv_channels, kernel_size=2, stride=2, padding=0
        )
        self.deconv_relu = nn.ReLU(inplace=True)

        # Final prediction (one mask per class)
        self.predictor = nn.Conv2d(conv_channels, num_classes, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, roi_features):
        """
        Args:
            roi_features: [num_rois, 256, 14, 14]

        Returns:
            mask_logits: [num_rois, num_classes, 28, 28]
        """
        x = self.conv_layers(roi_features)
        x = self.deconv(x)
        x = self.deconv_relu(x)
        x = self.predictor(x)

        return x


if __name__ == "__main__":
    print("Testing Mask Head...")

    mask_head = MaskHead(
        in_channels=256,
        num_classes=2,
        output_size=28,
    )

    # Dummy RoI features
    roi_features = torch.randn(5, 256, 14, 14)

    # Forward pass
    mask_logits = mask_head(roi_features)

    print(f"\nInput: {roi_features.shape}")
    print(f"Output: {mask_logits.shape}")
    print(f"Expected: [5, 2, 28, 28]")

    assert mask_logits.shape == (5, 2, 28, 28)

    # Test sigmoid activation
    masks_prob = torch.sigmoid(mask_logits)
    print(f"Mask probabilities range: [{masks_prob.min():.3f}, {masks_prob.max():.3f}]")

    print("\n✓ Mask Head test passed!")
