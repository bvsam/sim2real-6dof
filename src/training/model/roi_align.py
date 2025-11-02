"""
RoI Align operation for extracting fixed-size features from proposals.
Uses torchvision's built-in RoIAlign with pyramid level assignment.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.ops import RoIAlign as TorchvisionRoIAlign


class PyramidRoIAlign(nn.Module):
    """
    Multi-level RoI Align that assigns each RoI to the appropriate FPN level
    based on its size (similar to the original FPN paper and NOCS implementation).
    """

    def __init__(self, output_size, sampling_ratio=2):
        """
        Args:
            output_size: (height, width) of output features (e.g., 14 for NOCS head)
            sampling_ratio: Number of sampling points per RoI bin
        """
        super().__init__()

        self.output_size = (
            output_size
            if isinstance(output_size, tuple)
            else (output_size, output_size)
        )
        self.sampling_ratio = sampling_ratio

        # Create RoIAlign for each FPN level
        # P2: stride 4, P3: stride 8, P4: stride 16, P5: stride 32
        self.roi_align_p2 = TorchvisionRoIAlign(
            self.output_size, spatial_scale=1 / 4, sampling_ratio=sampling_ratio
        )
        self.roi_align_p3 = TorchvisionRoIAlign(
            self.output_size, spatial_scale=1 / 8, sampling_ratio=sampling_ratio
        )
        self.roi_align_p4 = TorchvisionRoIAlign(
            self.output_size, spatial_scale=1 / 16, sampling_ratio=sampling_ratio
        )
        self.roi_align_p5 = TorchvisionRoIAlign(
            self.output_size, spatial_scale=1 / 32, sampling_ratio=sampling_ratio
        )

        self.roi_aligns = [
            self.roi_align_p2,
            self.roi_align_p3,
            self.roi_align_p4,
            self.roi_align_p5,
        ]

    def assign_boxes_to_levels(
        self, boxes, image_size=(480, 640), canonical_box_size=224, canonical_level=4
    ):
        """
        Assign each box to an FPN level based on its size.

        Formula from FPN paper:
            level = floor(canonical_level + log2(sqrt(box_area) / canonical_box_size))

        Args:
            boxes: [N, 4] tensor of boxes (x1, y1, x2, y2)
            image_size: (H, W) of input image
            canonical_box_size: Reference box size (224 in FPN paper)
            canonical_level: Reference level (4 = P4 in FPN paper)

        Returns:
            levels: [N] tensor of level assignments (0=P2, 1=P3, 2=P4, 3=P5)
        """
        # Calculate box areas
        box_widths = boxes[:, 2] - boxes[:, 0]
        box_heights = boxes[:, 3] - boxes[:, 1]
        box_areas = box_widths * box_heights

        # Calculate image area for normalization (as in NOCS implementation)
        image_area = image_size[0] * image_size[1]

        # Assign to level based on box size
        # level = canonical_level + log2(sqrt(box_area) / canonical_box_size)
        target_levels = canonical_level + torch.log2(
            torch.sqrt(box_areas)
            / (
                canonical_box_size
                / torch.sqrt(torch.tensor(image_area, dtype=torch.float32))
            )
        )

        # Clamp to valid range [2, 5] then convert to [0, 3]
        target_levels = torch.clamp(target_levels.floor().long(), min=2, max=5) - 2

        return target_levels

    def forward(self, features, boxes, image_size=(480, 640)):
        """
        Args:
            features: OrderedDict of FPN features [p2, p3, p4, p5]
                Each feature: [B, 256, H, W]
            boxes: List of [num_boxes, 4] tensors (one per image in batch)
                Box format: (x1, y1, x2, y2) in absolute coordinates
            image_size: (H, W) of input image

        Returns:
            roi_features: [total_num_boxes, 256, output_size, output_size]
        """
        feature_list = list(features.values())  # [P2, P3, P4, P5]

        # Concatenate all boxes and track which image they belong to
        all_boxes = []
        box_to_image_idx = []

        for img_idx, boxes_per_image in enumerate(boxes):
            all_boxes.append(boxes_per_image)
            box_to_image_idx.append(
                torch.full(
                    (len(boxes_per_image),),
                    img_idx,
                    dtype=torch.int64,
                    device=boxes_per_image.device,
                )
            )

        all_boxes = torch.cat(all_boxes, dim=0)
        box_to_image_idx = torch.cat(box_to_image_idx, dim=0)

        # Assign boxes to FPN levels
        levels = self.assign_boxes_to_levels(all_boxes, image_size)

        # Prepare boxes with batch indices for RoIAlign
        # RoIAlign expects boxes in format: [batch_idx, x1, y1, x2, y2]
        boxes_with_indices = torch.cat(
            [box_to_image_idx.unsqueeze(1).float(), all_boxes], dim=1
        )

        # Extract features from each level
        pooled_features = []

        for level_idx in range(4):  # P2, P3, P4, P5
            # Find boxes assigned to this level
            level_mask = levels == level_idx

            if level_mask.sum() == 0:
                continue

            level_boxes = boxes_with_indices[level_mask]

            # Apply RoIAlign for this level
            roi_align = self.roi_aligns[level_idx]
            level_features = roi_align(feature_list[level_idx], level_boxes)

            pooled_features.append(
                (level_mask.nonzero(as_tuple=True)[0], level_features)
            )

        # Reorder features to match original box order
        output_features = torch.zeros(
            len(all_boxes),
            feature_list[0].shape[1],  # 256 channels
            self.output_size[0],
            self.output_size[1],
            device=all_boxes.device,
            dtype=feature_list[0].dtype,
        )

        for indices, features in pooled_features:
            output_features[indices] = features

        return output_features


if __name__ == "__main__":
    print("Testing PyramidRoIAlign...")

    from collections import OrderedDict

    # Create dummy FPN features
    features = OrderedDict(
        [
            ("p2", torch.randn(2, 256, 120, 160)),
            ("p3", torch.randn(2, 256, 60, 80)),
            ("p4", torch.randn(2, 256, 30, 40)),
            ("p5", torch.randn(2, 256, 15, 20)),
        ]
    )

    # Create dummy proposals
    # Image 0: 3 boxes, Image 1: 2 boxes
    boxes = [
        torch.tensor(
            [
                [50, 60, 150, 200],  # medium box
                [10, 10, 30, 30],  # small box
                [200, 100, 500, 400],  # large box
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [100, 100, 200, 250],
                [300, 200, 450, 380],
            ],
            dtype=torch.float32,
        ),
    ]

    roi_align = PyramidRoIAlign(output_size=14)
    roi_features = roi_align(features, boxes, image_size=(480, 640))

    print(f"\nInput:")
    print(f"  Batch size: 2")
    print(f"  Boxes per image: [3, 2]")
    print(f"  Total boxes: 5")

    print(f"\nOutput:")
    print(f"  RoI features shape: {roi_features.shape}")
    print(f"  Expected: [5, 256, 14, 14]")

    assert roi_features.shape == (5, 256, 14, 14), "Shape mismatch!"

    print("\n✓ PyramidRoIAlign test passed!")
