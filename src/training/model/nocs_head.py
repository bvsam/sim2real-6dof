"""
NOCS Coordinate Prediction Head.

Predicts NOCS (Normalized Object Coordinate Space) maps using classification.
Each coordinate value [0, 1] is discretized into bins and predicted via softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NOCSHead(nn.Module):
    """
    NOCS prediction head that outputs 3D coordinate maps.

    Architecture (from NOCS paper):
    - Input: 14x14x256 RoI features
    - 8x Conv2D (3x3, 512 channels) at 14x14
    - 1x ConvTranspose2D (2x2, stride=2) -> 28x28
    - 1x Conv2D (1x1) -> final prediction
    - Output: 28x28 x (num_classes * num_bins * 3)

    Uses classification (not regression) with 32 bins per coordinate.
    """

    def __init__(
        self,
        in_channels=256,
        num_conv_layers=8,
        conv_channels=512,
        num_classes=2,  # background + mug (expandable for multi-class)
        num_bins=32,
        output_size=28,
        use_bn=False,
    ):
        """
        Args:
            in_channels: Input channels from RoI features (256 from FPN)
            num_conv_layers: Number of 3x3 conv layers (8 in paper)
            conv_channels: Channels in conv layers (512 in paper)
            num_classes: Number of object classes (including background)
            num_bins: Number of bins for discretizing [0, 1] range (32 in paper)
            output_size: Output spatial size (28 in paper)
            use_bn: Whether to use batch normalization
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_bins = num_bins
        self.output_size = output_size

        # Convolutional layers at 14x14 resolution
        conv_layers = []
        for i in range(num_conv_layers):
            in_ch = in_channels if i == 0 else conv_channels
            conv_layers.append(
                nn.Conv2d(in_ch, conv_channels, kernel_size=3, padding=1)
            )

            if use_bn:
                conv_layers.append(nn.BatchNorm2d(conv_channels))

            conv_layers.append(nn.ReLU(inplace=True))

        self.conv_layers = nn.Sequential(*conv_layers)

        # Upsampling layer: 14x14 -> 28x28
        self.deconv = nn.ConvTranspose2d(
            conv_channels, conv_channels, kernel_size=2, stride=2, padding=0
        )
        self.deconv_relu = nn.ReLU(inplace=True)

        # Final prediction layer
        # Output: num_classes * 3 (xyz) * num_bins
        self.predictor = nn.Conv2d(
            conv_channels,
            num_classes * 3 * num_bins,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, roi_features):
        """
        Args:
            roi_features: [num_rois, 256, 14, 14] features from RoI Align

        Returns:
            nocs_logits: [num_rois, num_classes, 3, num_bins, 28, 28]
                Logits for each coordinate bin (before softmax)
        """
        # Apply conv layers
        x = self.conv_layers(roi_features)

        # Upsample to 28x28
        x = self.deconv(x)
        x = self.deconv_relu(x)

        # Final prediction
        x = self.predictor(x)  # [N, num_classes*3*num_bins, 28, 28]

        # Reshape to separate dimensions
        # [N, num_classes*3*num_bins, 28, 28] -> [N, num_classes, 3, num_bins, 28, 28]
        N = x.shape[0]
        x = x.view(
            N, self.num_classes, 3, self.num_bins, self.output_size, self.output_size
        )

        return x

    def decode_predictions(self, nocs_logits, class_ids=None):
        """
        Convert bin classification logits to coordinate values.

        Args:
            nocs_logits: [num_rois, num_classes, 3, num_bins, 28, 28]
            class_ids: [num_rois] class ID for each RoI (if None, uses argmax)

        Returns:
            nocs_coords: [num_rois, 3, 28, 28] coordinate maps in [0, 1]
        """
        N = nocs_logits.shape[0]

        # Apply softmax over bins
        probs = F.softmax(nocs_logits, dim=3)  # [N, num_classes, 3, num_bins, 28, 28]

        # Get bin centers [0, 1, 2, ..., 31] -> [0.015625, 0.046875, ..., 0.984375]
        bin_centers = (
            torch.arange(self.num_bins, device=nocs_logits.device).float() + 0.5
        ) / self.num_bins
        bin_centers = bin_centers.view(1, 1, 1, self.num_bins, 1, 1)

        # Weighted sum over bins to get expected coordinate value
        coords_all_classes = (probs * bin_centers).sum(
            dim=3
        )  # [N, num_classes, 3, 28, 28]

        # Select coordinates for the predicted/given class
        if class_ids is None:
            # During inference without class info, take max prob class
            # For single-class (mug only), just take class 1
            coords = coords_all_classes[:, 1]  # [N, 3, 28, 28]
        else:
            # During training/evaluation with known class IDs
            coords = torch.stack(
                [coords_all_classes[i, class_ids[i]] for i in range(N)]
            )  # [N, 3, 28, 28]

        return coords


class NOCSClassificationHead(nn.Module):
    """
    Combined head for object classification and NOCS prediction.
    """

    def __init__(
        self,
        in_channels=256,
        num_classes=2,
        num_bins=32,
        roi_size=14,
        nocs_output_size=28,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Classification head (simple MLP)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_cls = nn.Linear(in_channels, num_classes)

        # Bounding box regression head (standard Faster R-CNN)
        self.fc_bbox = nn.Sequential(
            nn.Linear(in_channels * roi_size * roi_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes * 4),
        )

        # NOCS prediction head
        self.nocs_head = NOCSHead(
            in_channels=in_channels,
            num_classes=num_classes,
            num_bins=num_bins,
            output_size=nocs_output_size,
        )

    def forward(self, roi_features):
        """
        Args:
            roi_features: [num_rois, 256, 14, 14]

        Returns:
            class_logits: [num_rois, num_classes]
            bbox_deltas: [num_rois, num_classes*4]
            nocs_logits: [num_rois, num_classes, 3, num_bins, 28, 28]
        """
        N = roi_features.shape[0]

        # Classification
        pooled_features = self.avgpool(roi_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        class_logits = self.fc_cls(pooled_features)

        # Bounding box regression
        flat_features = roi_features.view(N, -1)
        bbox_deltas = self.fc_bbox(flat_features)

        # NOCS prediction
        nocs_logits = self.nocs_head(roi_features)

        return class_logits, bbox_deltas, nocs_logits


if __name__ == "__main__":
    print("Testing NOCS Head...")

    # Test NOCS head alone
    nocs_head = NOCSHead(
        in_channels=256,
        num_classes=2,
        num_bins=32,
        output_size=28,
    )

    # Dummy RoI features (5 RoIs from previous test)
    roi_features = torch.randn(5, 256, 14, 14)

    # Forward pass
    nocs_logits = nocs_head(roi_features)

    print(f"\nNOCS Head:")
    print(f"  Input: {roi_features.shape}")
    print(f"  Output logits: {nocs_logits.shape}")
    print(f"  Expected: [5, 2, 3, 32, 28, 28]")
    assert nocs_logits.shape == (5, 2, 3, 32, 28, 28)

    # Test decoding
    coords = nocs_head.decode_predictions(
        nocs_logits, class_ids=torch.ones(5, dtype=torch.long)
    )
    print(f"  Decoded coords: {coords.shape}")
    print(f"  Coord range: [{coords.min():.3f}, {coords.max():.3f}]")
    assert coords.shape == (5, 3, 28, 28)
    assert coords.min() >= 0.0 and coords.max() <= 1.0

    # Test combined head
    print("\nCombined Classification + NOCS Head:")
    combined_head = NOCSClassificationHead(
        in_channels=256,
        num_classes=2,
        num_bins=32,
        roi_size=14,
        nocs_output_size=28,
    )

    class_logits, bbox_deltas, nocs_logits = combined_head(roi_features)

    print(f"  Class logits: {class_logits.shape}")
    print(f"  BBox deltas: {bbox_deltas.shape}")
    print(f"  NOCS logits: {nocs_logits.shape}")

    assert class_logits.shape == (5, 2)
    assert bbox_deltas.shape == (5, 8)  # 2 classes * 4 coords
    assert nocs_logits.shape == (5, 2, 3, 32, 28, 28)

    print("\n✓ NOCS Head test passed!")
