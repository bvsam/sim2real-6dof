"""
Region Proposal Network (RPN) for NOCS R-CNN.
Generates object proposals from FPN feature maps.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


class RPNHead(nn.Module):
    """
    RPN head that predicts objectness scores and bounding box deltas.
    Applied to each FPN level independently.
    """

    def __init__(self, in_channels=256, num_anchors=3):
        """
        Args:
            in_channels: Number of input channels from FPN (256)
            num_anchors: Number of anchor boxes per spatial location (3)
        """
        super().__init__()

        # 3x3 conv for feature transformation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # 1x1 convs for predictions
        self.objectness = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_deltas = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        # Initialize weights
        for layer in [self.conv, self.objectness, self.bbox_deltas]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Feature map [B, 256, H, W]

        Returns:
            objectness: [B, num_anchors, H, W] - objectness scores
            bbox_deltas: [B, num_anchors*4, H, W] - bounding box deltas
        """
        x = F.relu(self.conv(x))
        objectness = self.objectness(x)
        bbox_deltas = self.bbox_deltas(x)

        return objectness, bbox_deltas


class AnchorGenerator:
    """
    Generates anchor boxes at multiple scales and aspect ratios.
    """

    def __init__(
        self,
        sizes=((32,), (64,), (128,), (256,)),  # For P2, P3, P4, P5
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
        strides=(4, 8, 16, 32),
    ):
        """
        Args:
            sizes: Anchor sizes for each FPN level
            aspect_ratios: Aspect ratios for anchors
            strides: Feature map strides for each level
        """
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.cell_anchors = self._generate_cell_anchors()

    def _generate_cell_anchors(self):
        """Generate base anchor boxes for each FPN level."""
        cell_anchors = []

        for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios):
            anchors = []
            for size in sizes:
                area = size**2
                for aspect_ratio in aspect_ratios:
                    w = np.sqrt(area / aspect_ratio)
                    h = w * aspect_ratio

                    # Anchor format: [x1, y1, x2, y2] centered at origin
                    anchor = [-w / 2, -h / 2, w / 2, h / 2]
                    anchors.append(anchor)

            cell_anchors.append(torch.tensor(anchors, dtype=torch.float32))

        return cell_anchors

    def generate_anchors(self, feature_shapes, image_size, device):
        """
        Generate anchors for all FPN levels.

        Args:
            feature_shapes: List of (H, W) for each FPN level
            image_size: (H, W) of input image
            device: torch device

        Returns:
            anchors: [num_total_anchors, 4] tensor of anchor boxes
        """
        all_anchors = []

        for level, (feat_h, feat_w) in enumerate(feature_shapes):
            stride = self.strides[level]
            cell_anchors = self.cell_anchors[level].to(device)

            # Generate grid of anchor centers
            shift_x = torch.arange(0, feat_w, device=device) * stride
            shift_y = torch.arange(0, feat_h, device=device) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=2).reshape(
                -1, 4
            )

            # Apply shifts to cell anchors
            anchors = (shifts.view(-1, 1, 4) + cell_anchors.view(1, -1, 4)).reshape(
                -1, 4
            )
            all_anchors.append(anchors)

        return torch.cat(all_anchors, dim=0)


class RPN(nn.Module):
    """
    Region Proposal Network that generates object proposals.
    """

    def __init__(
        self,
        in_channels=256,
        num_anchors=3,
        nms_threshold=0.7,
        pre_nms_top_n_train=2000,
        pre_nms_top_n_test=1000,
        post_nms_top_n_train=2000,
        post_nms_top_n_test=1000,
        min_size=16,
    ):
        super().__init__()

        self.rpn_head = RPNHead(in_channels, num_anchors)
        self.anchor_generator = AnchorGenerator()

        self.nms_threshold = nms_threshold
        self.pre_nms_top_n_train = pre_nms_top_n_train
        self.pre_nms_top_n_test = pre_nms_top_n_test
        self.post_nms_top_n_train = post_nms_top_n_train
        self.post_nms_top_n_test = post_nms_top_n_test
        self.min_size = min_size

    def forward(self, features, image_shapes):
        """
        Args:
            features: OrderedDict of FPN features [p2, p3, p4, p5]
            image_shapes: List of (H, W) for each image in batch

        Returns:
            proposals: List of [num_proposals, 4] tensors (one per image)
            objectness_scores: List of [num_proposals] tensors
        """
        # Get predictions from each FPN level
        objectness_scores = []
        bbox_deltas = []

        for feat_map in features.values():
            obj, deltas = self.rpn_head(feat_map)
            objectness_scores.append(obj)
            bbox_deltas.append(deltas)

        # Generate anchors
        feature_shapes = [feat.shape[-2:] for feat in features.values()]
        anchors = self.anchor_generator.generate_anchors(
            feature_shapes, image_shapes[0], objectness_scores[0].device
        )

        # Decode proposals for each image
        batch_size = objectness_scores[0].shape[0]
        proposals_per_image = []
        scores_per_image = []

        for img_idx in range(batch_size):
            # Concatenate predictions from all levels
            obj_scores = []
            deltas = []

            for level_obj, level_deltas in zip(objectness_scores, bbox_deltas):
                B, A, H, W = level_obj.shape
                obj_scores.append(level_obj[img_idx].permute(1, 2, 0).reshape(-1))
                deltas.append(level_deltas[img_idx].permute(1, 2, 0).reshape(-1, 4))

            obj_scores = torch.cat(obj_scores)
            deltas = torch.cat(deltas, dim=0)

            # Apply deltas to anchors to get proposals
            proposals = self._decode_boxes(anchors, deltas)

            # Clip to image boundaries
            proposals = self._clip_boxes(proposals, image_shapes[img_idx])

            # Remove small boxes
            keep = self._filter_boxes(proposals, self.min_size)
            proposals = proposals[keep]
            obj_scores = obj_scores[keep]

            # Sort by objectness score
            top_n = (
                self.pre_nms_top_n_train if self.training else self.pre_nms_top_n_test
            )
            obj_scores, top_idx = obj_scores.topk(min(top_n, len(obj_scores)))
            proposals = proposals[top_idx]

            # Apply NMS
            keep = nms(proposals, obj_scores, self.nms_threshold)
            post_nms_top_n = (
                self.post_nms_top_n_train if self.training else self.post_nms_top_n_test
            )
            keep = keep[:post_nms_top_n]

            proposals_per_image.append(proposals[keep])
            scores_per_image.append(obj_scores[keep])

        return proposals_per_image, scores_per_image

    @staticmethod
    def _decode_boxes(anchors, deltas):
        """Decode bounding box deltas to absolute coordinates."""
        # anchors: [N, 4] (x1, y1, x2, y2)
        # deltas: [N, 4] (dx, dy, dw, dh)

        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes

    @staticmethod
    def _clip_boxes(boxes, image_shape):
        """Clip boxes to image boundaries."""
        h, w = image_shape
        boxes[:, 0].clamp_(min=0, max=w)
        boxes[:, 1].clamp_(min=0, max=h)
        boxes[:, 2].clamp_(min=0, max=w)
        boxes[:, 3].clamp_(min=0, max=h)
        return boxes

    @staticmethod
    def _filter_boxes(boxes, min_size):
        """Remove boxes smaller than min_size."""
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep = (widths >= min_size) & (heights >= min_size)
        return torch.where(keep)[0]


if __name__ == "__main__":
    print("Testing RPN...")

    # Create dummy FPN features
    from collections import OrderedDict

    features = OrderedDict(
        [
            ("p2", torch.randn(2, 256, 120, 160)),
            ("p3", torch.randn(2, 256, 60, 80)),
            ("p4", torch.randn(2, 256, 30, 40)),
            ("p5", torch.randn(2, 256, 15, 20)),
        ]
    )

    image_shapes = [(480, 640), (480, 640)]

    rpn = RPN()
    rpn.eval()

    with torch.no_grad():
        proposals, scores = rpn(features, image_shapes)

    print(f"\nBatch size: {len(proposals)}")
    for i, (props, scrs) in enumerate(zip(proposals, scores)):
        print(
            f"  Image {i}: {props.shape[0]} proposals, scores range: [{scrs.min():.3f}, {scrs.max():.3f}]"
        )
        print(f"    Sample proposal: {props[0]}")

    print("\n✓ RPN test passed!")
