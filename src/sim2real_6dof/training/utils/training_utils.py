"""
Training utilities for NOCS R-CNN.

Key component: ProposalTargetMatcher
Assigns ground truth to proposals for supervised training.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def batch_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] boxes in (x1, y1, x2, y2) format
        boxes2: [M, 4] boxes in (x1, y1, x2, y2) format

    Returns:
        iou: [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / union.clamp(min=1e-6)

    return iou


def encode_boxes(proposals: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Encode ground truth boxes relative to proposals (for bbox regression).

    Args:
        proposals: [N, 4] proposal boxes (x1, y1, x2, y2)
        gt_boxes: [N, 4] ground truth boxes (x1, y1, x2, y2)

    Returns:
        deltas: [N, 4] encoded deltas (dx, dy, dw, dh)
    """
    # Proposal box parameters
    prop_widths = proposals[:, 2] - proposals[:, 0]
    prop_heights = proposals[:, 3] - proposals[:, 1]
    prop_ctr_x = proposals[:, 0] + 0.5 * prop_widths
    prop_ctr_y = proposals[:, 1] + 0.5 * prop_heights

    # GT box parameters
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    # Encode
    dx = (gt_ctr_x - prop_ctr_x) / prop_widths
    dy = (gt_ctr_y - prop_ctr_y) / prop_heights
    dw = torch.log(gt_widths / prop_widths)
    dh = torch.log(gt_heights / prop_heights)

    deltas = torch.stack([dx, dy, dw, dh], dim=1)

    return deltas


def decode_boxes(proposals: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """
    Decode predicted deltas to get final boxes.

    Args:
        proposals: [N, 4] proposal boxes (x1, y1, x2, y2)
        deltas: [N, 4] predicted deltas (dx, dy, dw, dh)

    Returns:
        boxes: [N, 4] decoded boxes (x1, y1, x2, y2)
    """
    # Proposal box parameters
    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights

    # Decode
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    # Convert to (x1, y1, x2, y2)
    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


class ProposalTargetMatcher:
    """
    Matches proposals to ground truth for training.

    For each proposal, assigns:
    - Class label
    - Bounding box target
    - Mask target
    - NOCS coordinate target
    """

    def __init__(
        self,
        fg_iou_threshold=0.5,  # IoU threshold for positive samples
        bg_iou_threshold=0.5,  # IoU threshold for negative samples
        batch_size_per_image=512,  # RoIs per image
        positive_fraction=0.25,  # Fraction of positive RoIs
        mask_size=28,
        nocs_size=28,
    ):
        self.fg_iou_threshold = fg_iou_threshold
        self.bg_iou_threshold = bg_iou_threshold
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.mask_size = mask_size
        self.nocs_size = nocs_size

    def __call__(
        self,
        proposals: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_masks: List[torch.Tensor],
        gt_nocs: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Match proposals to ground truth.

        Args:
            proposals: List of [N, 4] proposal boxes per image
            gt_boxes: List of [M, 4] ground truth boxes per image
            gt_labels: List of [M] class labels per image
            gt_masks: List of [M, H, W] masks per image
            gt_nocs: List of [M, 3, H, W] NOCS maps per image

        Returns:
            matched_proposals: List of sampled proposals per image
            targets: Dictionary with matched targets:
                - labels: [total_rois] class labels
                - bbox_targets: [total_rois, 4] bbox regression targets
                - masks: [total_rois, mask_size, mask_size] mask targets
                - nocs_coords: [total_rois, 3, nocs_size, nocs_size] NOCS targets
        """
        matched_proposals = []
        all_labels = []
        all_bbox_targets = []
        all_masks = []
        all_nocs = []

        for props, gt_box, gt_label, gt_mask, gt_noc in zip(
            proposals, gt_boxes, gt_labels, gt_masks, gt_nocs
        ):
            # Compute IoU between proposals and GT
            if len(gt_box) == 0:
                # No ground truth - all proposals are background
                num_proposals = min(self.batch_size_per_image, len(props))
                sampled_props = props[:num_proposals]
                sampled_labels = torch.zeros(
                    num_proposals, dtype=torch.long, device=props.device
                )
                sampled_bbox_targets = torch.zeros(
                    (num_proposals, 4), device=props.device
                )
                sampled_masks = torch.zeros(
                    (num_proposals, self.mask_size, self.mask_size), device=props.device
                )
                sampled_nocs = torch.zeros(
                    (num_proposals, 3, self.nocs_size, self.nocs_size),
                    device=props.device,
                )
            else:
                iou_matrix = batch_iou(props, gt_box)  # [N_props, N_gt]

                # For each proposal, find best matching GT
                max_iou, matched_gt_idx = iou_matrix.max(dim=1)  # [N_props]

                # Assign labels based on IoU
                labels = gt_label[matched_gt_idx]  # [N_props]
                labels[max_iou < self.bg_iou_threshold] = 0  # Background

                # Sample positive and negative proposals
                positive_mask = labels > 0
                negative_mask = labels == 0

                num_positive = min(
                    int(self.batch_size_per_image * self.positive_fraction),
                    positive_mask.sum().item(),
                )
                num_negative = min(
                    self.batch_size_per_image - num_positive, negative_mask.sum().item()
                )

                # Sample indices
                positive_indices = torch.where(positive_mask)[0]
                negative_indices = torch.where(negative_mask)[0]

                if len(positive_indices) > num_positive:
                    perm = torch.randperm(len(positive_indices), device=props.device)
                    positive_indices = positive_indices[perm[:num_positive]]

                if len(negative_indices) > num_negative:
                    perm = torch.randperm(len(negative_indices), device=props.device)
                    negative_indices = negative_indices[perm[:num_negative]]

                sampled_indices = torch.cat([positive_indices, negative_indices])

                # Get matched targets
                sampled_props = props[sampled_indices]
                sampled_labels = labels[sampled_indices]
                sampled_matched_gt_idx = matched_gt_idx[sampled_indices]

                # Bbox targets (only for positive samples)
                sampled_bbox_targets = torch.zeros(
                    (len(sampled_props), 4), device=props.device
                )
                if len(positive_indices) > 0:
                    pos_props = sampled_props[: len(positive_indices)]
                    pos_gt_boxes = gt_box[
                        sampled_matched_gt_idx[: len(positive_indices)]
                    ]
                    sampled_bbox_targets[: len(positive_indices)] = encode_boxes(
                        pos_props, pos_gt_boxes
                    )

                # Mask and NOCS targets (only for positive samples)
                sampled_masks = torch.zeros(
                    (len(sampled_props), self.mask_size, self.mask_size),
                    device=props.device,
                )
                sampled_nocs = torch.zeros(
                    (len(sampled_props), 3, self.nocs_size, self.nocs_size),
                    device=props.device,
                )

                if len(positive_indices) > 0:
                    pos_gt_masks = gt_mask[
                        sampled_matched_gt_idx[: len(positive_indices)]
                    ]
                    pos_gt_nocs = gt_noc[
                        sampled_matched_gt_idx[: len(positive_indices)]
                    ]

                    # Crop and resize masks/NOCS to RoI
                    for i, (prop, mask, nocs) in enumerate(
                        zip(
                            sampled_props[: len(positive_indices)],
                            pos_gt_masks,
                            pos_gt_nocs,
                        )
                    ):
                        # Extract RoI region
                        x1, y1, x2, y2 = prop.long()
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(mask.shape[1], x2)
                        y2 = min(mask.shape[0], y2)

                        # Crop
                        mask_roi = mask[y1:y2, x1:x2]
                        nocs_roi = nocs[:, y1:y2, x1:x2]

                        # Resize to target size
                        if mask_roi.numel() > 0:
                            mask_resized = F.interpolate(
                                mask_roi.unsqueeze(0).unsqueeze(0).float(),
                                size=(self.mask_size, self.mask_size),
                                mode="nearest",
                            )[0, 0]
                            sampled_masks[i] = mask_resized

                        if nocs_roi.numel() > 0:
                            nocs_resized = F.interpolate(
                                nocs_roi.unsqueeze(0),
                                size=(self.nocs_size, self.nocs_size),
                                mode="bilinear",
                                align_corners=False,
                            )[0]
                            sampled_nocs[i] = nocs_resized

            matched_proposals.append(sampled_props)
            all_labels.append(sampled_labels)
            all_bbox_targets.append(sampled_bbox_targets)
            all_masks.append(sampled_masks)
            all_nocs.append(sampled_nocs)

        # Concatenate all batches
        targets = {
            "labels": torch.cat(all_labels, dim=0),
            "bbox_targets": torch.cat(all_bbox_targets, dim=0),
            "masks": torch.cat(all_masks, dim=0),
            "nocs_coords": torch.cat(all_nocs, dim=0),
        }

        return matched_proposals, targets


if __name__ == "__main__":
    print("Testing Training Utilities...")

    # Test IoU
    print("\n1. Testing IoU computation:")
    boxes1 = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
    boxes2 = torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
    iou = batch_iou(boxes1, boxes2)
    print(f"   IoU matrix:\n{iou}")
    print(f"   Expected: [[1.0, 0.0], [~0.14, 0.0]]")

    # Test box encoding/decoding
    print("\n2. Testing box encoding/decoding:")
    proposals = torch.tensor([[10, 10, 20, 20]], dtype=torch.float32)
    gt_boxes = torch.tensor([[12, 12, 22, 22]], dtype=torch.float32)
    deltas = encode_boxes(proposals, gt_boxes)
    decoded = decode_boxes(proposals, deltas)
    print(f"   Original GT: {gt_boxes}")
    print(f"   Encoded deltas: {deltas}")
    print(f"   Decoded boxes: {decoded}")
    assert torch.allclose(decoded, gt_boxes, atol=1e-5)

    # Test ProposalTargetMatcher
    print("\n3. Testing ProposalTargetMatcher:")
    matcher = ProposalTargetMatcher(
        batch_size_per_image=4,
        positive_fraction=0.5,
    )

    # Create dummy data
    proposals = [
        torch.tensor(
            [
                [10, 10, 50, 50],
                [60, 60, 100, 100],
                [5, 5, 15, 15],
                [200, 200, 250, 250],
            ],
            dtype=torch.float32,
        )
    ]

    gt_boxes = [torch.tensor([[12, 12, 48, 48]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]
    gt_masks = [torch.ones(1, 480, 640)]
    gt_nocs = [torch.rand(1, 3, 480, 640)]

    matched_props, targets = matcher(proposals, gt_boxes, gt_labels, gt_masks, gt_nocs)

    print(f"   Input proposals: {len(proposals[0])}")
    print(f"   Sampled proposals: {len(matched_props[0])}")
    print(f"   Labels: {targets['labels']}")
    print(f"   BBox targets shape: {targets['bbox_targets'].shape}")
    print(f"   Masks shape: {targets['masks'].shape}")
    print(f"   NOCS shape: {targets['nocs_coords'].shape}")

    assert len(matched_props[0]) == 4  # batch_size_per_image
    assert targets["labels"].shape[0] == 4

    print("\n✓ Training utilities test passed!")
