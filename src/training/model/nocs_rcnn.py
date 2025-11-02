"""
Main NOCS R-CNN Model.

Combines all components:
- ResNet-50 + FPN backbone
- Region Proposal Network (RPN)
- RoI Align
- Classification, BBox, Mask, and NOCS prediction heads
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import ResNetFPNBackbone
from .mask_head import MaskHead
from .nocs_head import NOCSHead
from .roi_align import PyramidRoIAlign
from .rpn import RPN


class NOCSRCNN(nn.Module):
    """
    NOCS R-CNN: Mask R-CNN with NOCS coordinate prediction.

    Architecture:
    1. Backbone: ResNet-50 + FPN
    2. RPN: Generate object proposals
    3. RoI Align: Extract fixed-size features for each proposal
    4. Heads:
       - Classification (object class)
       - Bounding box regression
       - Mask prediction
       - NOCS coordinate prediction (novel component)
    """

    def __init__(
        self,
        num_classes=2,  # background + mug
        num_bins=32,
        roi_output_size=14,
        mask_output_size=28,
        nocs_output_size=28,
        pretrained_backbone=True,
        # RPN parameters
        rpn_nms_threshold=0.7,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_bins = num_bins
        self.roi_output_size = roi_output_size

        # 1. Backbone
        self.backbone = ResNetFPNBackbone(pretrained=pretrained_backbone)

        # 2. RPN
        self.rpn = RPN(
            in_channels=256,
            num_anchors=3,
            nms_threshold=rpn_nms_threshold,
            pre_nms_top_n_train=rpn_pre_nms_top_n_train,
            pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            post_nms_top_n_train=rpn_post_nms_top_n_train,
            post_nms_top_n_test=rpn_post_nms_top_n_test,
        )

        # 3. RoI Align
        self.roi_align = PyramidRoIAlign(output_size=roi_output_size)

        # 4. Heads
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_cls = nn.Linear(256, num_classes)

        # Bounding box regression head
        self.fc_bbox = nn.Sequential(
            nn.Linear(256 * roi_output_size * roi_output_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes * 4),
        )

        # Mask head
        self.mask_head = MaskHead(
            in_channels=256,
            num_classes=num_classes,
            output_size=mask_output_size,
        )

        # NOCS head
        self.nocs_head = NOCSHead(
            in_channels=256,
            num_classes=num_classes,
            num_bins=num_bins,
            output_size=nocs_output_size,
        )

    def freeze_backbone_stages(self, stage):
        """
        Freeze backbone stages for progressive training.

        Stage schedule from NOCS paper:
        - Stage 1: Freeze all ResNet (stage=5), train heads/RPN/FPN
        - Stage 2: Freeze below C4 (stage=4)
        - Stage 3: Freeze below C3 (stage=3)
        """
        self.backbone.freeze_stages(stage)

    def forward(
        self,
        images: torch.Tensor,
        proposals: Optional[List[torch.Tensor]] = None,
        targets: Optional[List[Dict]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: [B, 3, H, W] input images
            proposals: Optional list of [N, 4] proposal boxes (for training)
                If None, uses RPN to generate proposals
            targets: Optional list of target dictionaries (for training)
                Each dict contains: boxes, labels, masks, nocs_coords

        Returns:
            Dictionary containing:
            - During training: all predictions + proposals
            - During inference: final detections
        """
        # 1. Backbone
        features = self.backbone(images)

        # 2. RPN
        image_shapes = [(img.shape[-2], img.shape[-1]) for img in images]

        if proposals is None:
            # Generate proposals using RPN
            proposals, proposal_scores = self.rpn(features, image_shapes)
        else:
            # Use provided proposals (during training after target matching)
            proposal_scores = None

        # 3. RoI Align
        if len(proposals) > 0 and proposals[0].shape[0] > 0:
            roi_features = self.roi_align(features, proposals, image_shapes[0])
        else:
            # No proposals - return empty predictions
            return self._empty_predictions(images.device)

        # 4. Prediction Heads
        N = roi_features.shape[0]

        # Classification
        pooled = self.avgpool(roi_features)
        pooled = pooled.view(pooled.size(0), -1)
        class_logits = self.fc_cls(pooled)

        # BBox regression
        bbox_features = roi_features.view(N, -1)
        bbox_deltas = self.fc_bbox(bbox_features)

        # Mask prediction
        mask_logits = self.mask_head(roi_features)

        # NOCS prediction
        nocs_logits = self.nocs_head(roi_features)

        # Package outputs
        outputs = {
            "class_logits": class_logits,
            "bbox_deltas": bbox_deltas,
            "mask_logits": mask_logits,
            "nocs_logits": nocs_logits,
            "proposals": proposals,
            "roi_features": roi_features,
        }

        if proposal_scores is not None:
            outputs["proposal_scores"] = proposal_scores

        return outputs

    def _empty_predictions(self, device):
        """Return empty predictions when there are no proposals."""
        return {
            "class_logits": torch.zeros((0, self.num_classes), device=device),
            "bbox_deltas": torch.zeros((0, self.num_classes * 4), device=device),
            "mask_logits": torch.zeros((0, self.num_classes, 28, 28), device=device),
            "nocs_logits": torch.zeros(
                (0, self.num_classes, 3, self.num_bins, 28, 28), device=device
            ),
            "proposals": [],
        }

    def predict(
        self,
        images: torch.Tensor,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Inference mode - generate final predictions.

        Args:
            images: [B, 3, H, W] input images
            score_threshold: Minimum confidence score
            nms_threshold: NMS IoU threshold

        Returns:
            List of prediction dicts (one per image) containing:
            - boxes: [N, 4] final bounding boxes
            - labels: [N] class labels
            - scores: [N] confidence scores
            - masks: [N, H, W] binary masks
            - nocs_coords: [N, 3, H, W] NOCS coordinate maps
        """
        self.eval()

        with torch.no_grad():
            outputs = self.forward(images)

            # TODO: Implement post-processing (NMS, score filtering, etc.)
            # For now, return raw outputs
            predictions = []

            # This will be implemented in the next step

        return predictions


if __name__ == "__main__":
    print("Testing NOCS R-CNN...")

    # Create model
    model = NOCSRCNN(
        num_classes=2,
        num_bins=32,
        pretrained_backbone=True,
    )

    print(f"\nModel created:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()

    batch_size = 2
    images = torch.randn(batch_size, 3, 480, 640)

    with torch.no_grad():
        outputs = model(images)

    print(f"\nOutputs:")
    print(f"  Class logits: {outputs['class_logits'].shape}")
    print(f"  BBox deltas: {outputs['bbox_deltas'].shape}")
    print(f"  Mask logits: {outputs['mask_logits'].shape}")
    print(f"  NOCS logits: {outputs['nocs_logits'].shape}")
    print(f"  Proposals per image: {[p.shape[0] for p in outputs['proposals']]}")

    # Test with manual proposals
    print("\nTesting with manual proposals...")
    proposals = [
        torch.tensor([[50, 60, 150, 200], [10, 10, 100, 100]], dtype=torch.float32),
        torch.tensor([[100, 100, 300, 400]], dtype=torch.float32),
    ]

    with torch.no_grad():
        outputs = model(images, proposals=proposals)

    print(f"  Total RoIs: {outputs['class_logits'].shape[0]}")
    print(f"  Expected: 3 (2 + 1)")
    assert outputs["class_logits"].shape[0] == 3

    # Test freezing
    print("\nTesting backbone freezing...")
    model.freeze_backbone_stages(5)  # Freeze all ResNet

    trainable_after_freeze = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"  Trainable after freezing: {trainable_after_freeze:,}")
    print(f"  Frozen: {total_params - trainable_after_freeze:,}")

    print("\n✓ NOCS R-CNN test passed!")
