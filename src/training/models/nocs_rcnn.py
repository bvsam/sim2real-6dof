"""
Custom ROI Heads that adds NOCS prediction to standard Mask R-CNN.
"""

from typing import Dict, List, Tuple

import torch
from detectron2.layers import ShapeSpec
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import Boxes, Instances
from torch import nn

from .nocs_head import NOCSHead, nocs_inference, nocs_loss


@ROI_HEADS_REGISTRY.register()
class NOCSROIHeads(StandardROIHeads):
    """
    Extended ROI heads that add NOCS coordinate prediction.

    Inherits from StandardROIHeads (which handles box, class, and mask prediction)
    and adds the NOCS prediction head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        # NOCS-specific config
        self.nocs_loss_weight = cfg.MODEL.NOCS_HEAD.LOSS_WEIGHT
        self.num_bins = cfg.MODEL.NOCS_HEAD.NUM_BINS

        # Create NOCS pooler (same as mask pooler typically)
        self.nocs_pooler = self._init_nocs_pooler(cfg, input_shape)

        # Create NOCS head
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        in_channels = input_shape[in_features[0]].channels  # 256 for FPN

        nocs_input_shape = ShapeSpec(
            channels=in_channels,
            height=cfg.MODEL.NOCS_HEAD.POOLER_RESOLUTION,
            width=cfg.MODEL.NOCS_HEAD.POOLER_RESOLUTION,
        )
        self.nocs_head = NOCSHead(cfg, nocs_input_shape)

    def _init_nocs_pooler(self, cfg, input_shape):
        """Initialize ROI pooler for NOCS head."""
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.NOCS_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.NOCS_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.NOCS_HEAD.POOLER_TYPE

        pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return pooler

    def forward(self, images, features, proposals, targets=None):
        """
        Forward pass for ROI heads.

        Args:
            images: ImageList
            features: Dict[str, Tensor] - FPN features (P2, P3, P4, P5)
            proposals: List[Instances] - RPN proposals
            targets: List[Instances] - Ground truth (only during training)

        Returns:
            During training: losses dict
            During inference: predictions (Instances)
        """
        if self.training:
            assert targets is not None
            proposals = self.label_and_sample_proposals(proposals, targets)

        del images

        if self.training:
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_nocs(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            # NOCS prediction is included in forward_with_given_boxes
            return pred_instances, {}

    def _forward_nocs(self, features, instances):
        """
        Forward pass for NOCS head during training.

        Args:
            features: Dict[str, Tensor] - FPN features
            instances: List[Instances] - Proposals with GT assignments

        Returns:
            Dict with NOCS losses
        """
        if not self.training:
            return {}

        # Only train on positive (foreground) proposals
        instances, _ = select_foreground_proposals(instances, self.num_classes)

        if len(instances) == 0:
            return {"loss_nocs": features[list(features.keys())[0]].sum() * 0}

        # Extract proposal boxes
        proposal_boxes = [x.proposal_boxes for x in instances]

        # Pool features
        features_list = [features[f] for f in self.in_features]
        nocs_features = self.nocs_pooler(features_list, proposal_boxes)

        # Predict NOCS
        nocs_pred = self.nocs_head(nocs_features)

        # Prepare ground truth
        gt_coords, gt_masks, gt_classes = self._prepare_nocs_targets(instances)

        # Compute loss
        loss, loss_dict = nocs_loss(
            nocs_pred, gt_coords, gt_masks, gt_classes, num_bins=self.num_bins
        )

        # Apply loss weight
        loss_dict = {k: v * self.nocs_loss_weight for k, v in loss_dict.items()}

        return loss_dict

    def _prepare_nocs_targets(self, instances):
        """
        Prepare ground truth NOCS targets from instances.

        Args:
            instances: List[Instances] with gt_coords and gt_masks

        Returns:
            gt_coords: [N, 28, 28, 3]
            gt_masks: [N, 28, 28]
            gt_classes: [N]
        """
        gt_coords = []
        gt_masks = []
        gt_classes = []

        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue

            # Stack across instances in this image
            gt_coords.append(instances_per_image.gt_coords)
            gt_masks.append(instances_per_image.gt_nocs_masks)
            gt_classes.append(instances_per_image.gt_classes)

        if len(gt_coords) == 0:
            device = instances[0].gt_boxes.device
            return (
                torch.zeros((0, 28, 28, 3), device=device),
                torch.zeros((0, 28, 28), device=device),
                torch.zeros((0,), dtype=torch.int64, device=device),
            )

        gt_coords = torch.cat(gt_coords, dim=0)
        gt_masks = torch.cat(gt_masks, dim=0)
        gt_classes = torch.cat(gt_classes, dim=0)

        return gt_coords, gt_masks, gt_classes

    def forward_with_given_boxes(self, features, instances):
        """
        Inference: predict masks and NOCS given detected boxes.

        Args:
            features: Dict[str, Tensor]
            instances: List[Instances] with pred_boxes and pred_classes

        Returns:
            List[Instances] with pred_masks and pred_coords added
        """
        # Standard mask prediction
        instances = super().forward_with_given_boxes(features, instances)

        # NOCS prediction
        instances = self._forward_nocs_inference(features, instances)

        return instances

    def _forward_nocs_inference(self, features, instances):
        """
        Predict NOCS coordinates during inference.

        Args:
            features: Dict[str, Tensor]
            instances: List[Instances] with pred_boxes and pred_classes

        Returns:
            List[Instances] with pred_coords added
        """
        # Collect all boxes
        pred_boxes = [x.pred_boxes for x in instances]

        # Pool features
        features_list = [features[f] for f in self.in_features]
        nocs_features = self.nocs_pooler(features_list, pred_boxes)

        # Predict NOCS logits
        nocs_pred = self.nocs_head(
            nocs_features
        )  # [N_total, num_classes, 3, num_bins, 28, 28]

        # Split predictions per image and convert to coordinates
        num_instances_per_image = [len(x) for x in instances]
        nocs_pred_per_image = nocs_pred.split(num_instances_per_image, dim=0)

        for instances_i, nocs_pred_i in zip(instances, nocs_pred_per_image):
            if len(instances_i) == 0:
                continue

            # Convert from bins to continuous coordinates
            pred_classes = instances_i.pred_classes
            coords = nocs_inference(nocs_pred_i, pred_classes)  # [N, 28, 28, 3]

            instances_i.pred_coords = coords

        return instances
