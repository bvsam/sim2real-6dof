"""
NOCS-enhanced Mask R-CNN using torchvision's pretrained model.

Adds NOCS coordinate prediction head to torchvision's MaskRCNN.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import (
    RoIHeads,
    fastrcnn_loss,
    maskrcnn_inference,
    maskrcnn_loss,
)
from torchvision.ops import MultiScaleRoIAlign

from .nocs_head import NOCSHead


class NOCSRoIHeads(RoIHeads):
    """
    Extended RoI heads with NOCS prediction capability.
    """

    def __init__(self, *args, nocs_roi_pool=None, nocs_head=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.nocs_roi_pool = nocs_roi_pool
        self.nocs_head = nocs_head

        # Storage for NOCS data to pass to external loss computation
        self.nocs_logits_storage = None
        self.nocs_labels_storage = None
        self.nocs_matched_idxs_storage = None
        self.nocs_proposals_storage = None

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Forward pass with NOCS prediction added.

        Follows the same pattern as the mask head from parent class.
        """
        # Validate targets
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if t["boxes"].dtype not in floating_point_types:
                    raise TypeError(
                        f"target boxes must of float type, instead got {t['boxes'].dtype}"
                    )
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(
                        f"target labels must of int64 type, instead got {t['labels'].dtype}"
                    )

        # Select training samples (or pass through for inference)
        if self.training:
            proposals, matched_idxs, labels, regression_targets = (
                self.select_training_samples(proposals, targets)
            )
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        # Box head
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: list[dict[str, torch.Tensor]] = []
        losses = {}

        # Box losses or postprocessing
        if self.training:
            if labels is None or regression_targets is None:
                raise ValueError("labels and regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        # Mask head (standard Mask R-CNN)
        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(
                    features, mask_proposals, image_shapes
                )
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError(
                        "targets, pos_matched_idxs, mask_logits cannot be None when training"
                    )

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
                )
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # NOCS head (our addition - follows mask head pattern)
        if self.nocs_roi_pool is not None and self.nocs_head is not None:
            nocs_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # During training, only focus on positive boxes (same as mask)
                num_images = len(proposals)
                nocs_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    nocs_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            # NOCS feature extraction and prediction
            nocs_features = self.nocs_roi_pool(features, nocs_proposals, image_shapes)
            nocs_logits = self.nocs_head(nocs_features)

            # Store for external NOCS loss computation
            if self.training:
                self.nocs_logits_storage = nocs_logits
                self.nocs_labels_storage = labels
                self.nocs_matched_idxs_storage = pos_matched_idxs
                self.nocs_proposals_storage = nocs_proposals
            else:
                # Add NOCS to results
                num_boxes = [len(r["boxes"]) for r in result]
                nocs_logits_split = nocs_logits.split(num_boxes, 0)
                for img_nocs, r in zip(nocs_logits_split, result):
                    r["nocs_logits"] = img_nocs

        return result, losses


class NOCSMaskRCNN(nn.Module):
    """
    NOCS-enhanced Mask R-CNN.

    Uses torchvision's pretrained MaskRCNN and adds a NOCS prediction head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        num_bins: int = 32,
        nocs_output_size: int = 28,
        pretrained: bool = True,
        pretrained_backbone: bool = True,
        # trainable_backbone_layers: int = 3,
    ):
        """
        Args:
            num_classes: Number of classes (including background)
            num_bins: Number of bins for NOCS discretization
            nocs_output_size: Output size for NOCS maps (28x28)
            pretrained: Load COCO-pretrained weights for entire model
            pretrained_backbone: Load ImageNet-pretrained weights for backbone
            # trainable_backbone_layers: Number of trainable backbone layers (0-5)
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_bins = num_bins

        # Load base Mask R-CNN
        if pretrained:
            weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
            self.model = maskrcnn_resnet50_fpn(
                weights=weights,
                # trainable_backbone_layers=trainable_backbone_layers,
            )
        else:
            weights_backbone = (
                ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            )
            self.model = maskrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=weights_backbone,
                # trainable_backbone_layers=trainable_backbone_layers,
            )

        # Replace the box predictor for our number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace the mask predictor for our number of classes
        mask_predictor_in_channels = (
            self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        )
        mask_dim_reduced = (
            self.model.roi_heads.mask_predictor.mask_fcn_logits.in_channels
        )
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            mask_predictor_in_channels, mask_dim_reduced, num_classes
        )

        # Create NOCS RoI pooler (same as mask pooler - 14x14 output)
        nocs_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],  # P2, P3, P4, P5
            output_size=14,  # Pool to 14x14 (input to NOCS head)
            sampling_ratio=2,
        )

        # Add NOCS prediction head
        nocs_head = NOCSHead(
            in_channels=self.model.backbone.out_channels,
            num_classes=num_classes,
            num_bins=num_bins,
            output_size=nocs_output_size,
        )

        # Replace RoI heads with our extended version
        old_roi_heads = self.model.roi_heads
        self.model.roi_heads = NOCSRoIHeads(
            box_roi_pool=old_roi_heads.box_roi_pool,
            box_head=old_roi_heads.box_head,
            box_predictor=old_roi_heads.box_predictor,
            fg_iou_thresh=old_roi_heads.proposal_matcher.high_threshold,
            bg_iou_thresh=old_roi_heads.proposal_matcher.low_threshold,
            batch_size_per_image=old_roi_heads.fg_bg_sampler.batch_size_per_image,
            positive_fraction=old_roi_heads.fg_bg_sampler.positive_fraction,
            bbox_reg_weights=old_roi_heads.box_coder.weights,
            score_thresh=old_roi_heads.score_thresh,
            nms_thresh=old_roi_heads.nms_thresh,
            detections_per_img=old_roi_heads.detections_per_img,
            mask_roi_pool=old_roi_heads.mask_roi_pool,
            mask_head=old_roi_heads.mask_head,
            mask_predictor=old_roi_heads.mask_predictor,
            nocs_roi_pool=nocs_roi_pool,
            nocs_head=nocs_head,
        )

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: List of [C, H, W] tensors (torchvision format)
            targets: Optional list of target dicts for training

        Returns:
            If training: dict with 'losses', 'nocs_logits', 'proposals'
            If inference: dict with 'detections' (includes NOCS predictions)
        """
        if self.training:
            # Training mode
            assert targets is not None, "Targets required during training"

            # Call torchvision's model
            loss_dict = self.model(images, targets)

            # Get NOCS logits and proposals from storage
            nocs_logits = self.model.roi_heads.nocs_logits_storage

            return {
                "losses": loss_dict,
                "nocs_logits": nocs_logits,
            }
        else:
            # Inference mode
            detections = self.model(images)

            return {
                "detections": detections,
            }

    def freeze_backbone_stages(self, stage: int):
        """
        Freeze backbone stages for progressive training.

        Args:
            stage: 0-5 (0 = nothing frozen, 5 = all backbone frozen)
        """
        backbone = self.model.backbone.body

        # Stage 5: Freeze entire backbone
        if stage >= 5:
            for param in backbone.parameters():
                param.requires_grad = False
            return

        # Freeze bottom-up pyramid layers
        layers_to_freeze = {
            0: [],
            1: ["conv1", "bn1"],
            2: ["conv1", "bn1", "layer1"],
            3: ["conv1", "bn1", "layer1", "layer2"],
            4: ["conv1", "bn1", "layer1", "layer2", "layer3"],
        }

        for layer_name in layers_to_freeze.get(stage, []):
            if hasattr(backbone, layer_name):
                layer = getattr(backbone, layer_name)
                for param in layer.parameters():
                    param.requires_grad = False


if __name__ == "__main__":
    print("Testing NOCS-enhanced Mask R-CNN...")

    # Create model
    model = NOCSMaskRCNN(
        num_classes=2,
        num_bins=32,
        pretrained=True,
    )

    print(f"\nModel created:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test training mode
    print("\nTesting training mode...")
    model.train()

    # Prepare inputs
    images = [
        torch.randn(3, 480, 640),
        torch.randn(3, 480, 640),
    ]

    targets = [
        {
            "boxes": torch.tensor([[50, 60, 150, 200]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "masks": torch.randint(0, 2, (1, 480, 640), dtype=torch.uint8),
        },
        {
            "boxes": torch.tensor([[100, 100, 300, 400]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "masks": torch.randint(0, 2, (1, 480, 640), dtype=torch.uint8),
        },
    ]

    # Forward pass
    outputs = model(images, targets)

    print(f"\nTraining outputs:")
    print(f"  Losses: {list(outputs['losses'].keys())}")
    if outputs["nocs_logits"] is not None:
        print(f"  NOCS logits shape: {outputs['nocs_logits'].shape}")

    # Test inference mode
    print("\nTesting inference mode...")
    model.eval()

    with torch.no_grad():
        outputs = model(images)

    detections = outputs["detections"]
    print(f"\nInference outputs:")
    print(f"  Number of images: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"  Image {i}:")
        print(f"    Boxes: {det['boxes'].shape}")
        print(f"    Labels: {det['labels'].shape}")
        print(f"    Scores: {det['scores'].shape}")
        print(f"    Masks: {det['masks'].shape}")
        if "nocs_logits" in det:
            print(f"    NOCS logits: {det['nocs_logits'].shape}")

    # Test freezing
    print("\nTesting backbone freezing...")
    model.freeze_backbone_stages(5)

    trainable_after_freeze = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"  Trainable after freezing: {trainable_after_freeze:,}")
    print(f"  Frozen: {total_params - trainable_after_freeze:,}")

    print("\n✓ NOCS-enhanced Mask R-CNN test passed!")
    print(f"  Trainable after freezing: {trainable_after_freeze:,}")
    print(f"  Frozen: {total_params - trainable_after_freeze:,}")

    print("\n✓ NOCS-enhanced Mask R-CNN test passed!")
    print(f"  Frozen: {total_params - trainable_after_freeze:,}")

    print("\n✓ NOCS-enhanced Mask R-CNN test passed!")
