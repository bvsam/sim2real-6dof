"""
Loss functions for NOCS R-CNN training.

Key insight from the paper: NOCS coordinates are predicted via
CLASSIFICATION (binning), not regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


def project_nocs_on_boxes(gt_nocs, boxes, matched_idxs, M):
    """
    Given NOCS coordinate maps and bounding boxes, crop and resize
    the NOCS maps to the proposal regions.

    Args:
        gt_nocs: [N_gt, 3, H, W] ground truth NOCS maps
        boxes: [N_proposals, 4] proposal boxes
        matched_idxs: [N_proposals] indices of matched GT for each proposal
        M: int, output size (e.g., 28)

    Returns:
        [N_proposals, 3, M, M] cropped and resized NOCS maps
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_nocs = gt_nocs.to(rois)
    return roi_align(gt_nocs, rois, (M, M), 1.0, aligned=True)


def nocs_loss(nocs_logits, proposals, gt_nocs, gt_labels, nocs_matched_idxs):
    """
    Compute NOCS coordinate loss using bin classification.

    Follows the same pattern as maskrcnn_loss from torchvision.

    Args:
        nocs_logits: [N_proposals, num_classes, 3, num_bins, M, M] predicted logits
        proposals: List of [N_proposals_i, 4] proposal boxes per image
        gt_nocs: List of [N_gt_i, 3, H, W] ground truth NOCS maps per image
        gt_labels: List of [N_gt_i] class labels per image
        nocs_matched_idxs: List of [N_proposals_i] matched GT indices per image

    Returns:
        loss: Scalar tensor
    """
    num_bins = nocs_logits.shape[3]
    discretization_size = nocs_logits.shape[-1]

    # Get labels for each positive proposal
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, nocs_matched_idxs)]

    # Project GT NOCS onto proposal boxes
    nocs_targets = [
        project_nocs_on_boxes(nocs, props, idxs, discretization_size)
        for nocs, props, idxs in zip(gt_nocs, proposals, nocs_matched_idxs)
    ]

    # Concatenate across batch
    labels = torch.cat(labels, dim=0)
    nocs_targets = torch.cat(nocs_targets, dim=0)  # [N, 3, M, M]

    # Handle empty case
    if nocs_targets.numel() == 0:
        return nocs_logits.sum() * 0

    # Convert continuous NOCS [0, 1] to bin indices [0, num_bins-1]
    nocs_targets_bins = (nocs_targets * num_bins).long()
    nocs_targets_bins = torch.clamp(nocs_targets_bins, 0, num_bins - 1)  # [N, 3, M, M]

    # Select predictions for correct class
    # nocs_logits: [N, num_classes, 3, num_bins, M, M]
    # We need: [N, 3, num_bins, M, M]
    class_specific_logits = nocs_logits[
        torch.arange(labels.shape[0], device=labels.device), labels
    ]  # [N, 3, num_bins, M, M]

    # Compute cross-entropy loss for each coordinate
    losses = []
    for coord_idx in range(3):  # x, y, z
        coord_logits = class_specific_logits[:, coord_idx]  # [N, num_bins, M, M]
        coord_targets = nocs_targets_bins[:, coord_idx]  # [N, M, M]

        # Reshape for cross_entropy: [N*M*M, num_bins] and [N*M*M]
        coord_logits_flat = coord_logits.permute(0, 2, 3, 1).reshape(-1, num_bins)
        coord_targets_flat = coord_targets.reshape(-1)

        loss_coord = F.cross_entropy(
            coord_logits_flat, coord_targets_flat, reduction="mean"
        )
        losses.append(loss_coord)

    # Average over x, y, z
    total_loss = sum(losses) / 3.0

    return total_loss


# Keep existing classes for backward compatibility
def compute_nocs_bin_targets(nocs_coords_continuous, num_bins=32):
    """Convert continuous NOCS coordinates [0, 1] to bin indices."""
    coords = torch.clamp(nocs_coords_continuous, 0.0, 1.0)
    bin_indices = (coords * num_bins).long()
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
    return bin_indices


class NOCSLoss(nn.Module):
    """Legacy wrapper - use nocs_loss function instead."""

    def __init__(self, num_bins=32):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, nocs_logits, target_coords, target_masks, class_ids):
        # This is the old API - keeping for compatibility but not used in new training
        raise NotImplementedError("Use nocs_loss function instead")


class MaskLoss(nn.Module):
    """Binary cross-entropy loss for mask prediction."""

    def __init__(self):
        super().__init__()

    def forward(self, mask_logits, target_masks, class_ids):
        # Not used with torchvision model (it computes mask loss internally)
        raise NotImplementedError("Mask loss computed internally by torchvision model")


class ClassificationLoss(nn.Module):
    """Cross-entropy loss for object classification."""

    def __init__(self):
        super().__init__()

    def forward(self, class_logits, target_class_ids):
        # Not used with torchvision model (it computes classification loss internally)
        raise NotImplementedError(
            "Classification loss computed internally by torchvision model"
        )


class BBoxLoss(nn.Module):
    """Smooth L1 loss for bounding box regression."""

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, bbox_deltas, target_deltas, class_ids, num_classes):
        # Not used with torchvision model (it computes bbox loss internally)
        raise NotImplementedError("BBox loss computed internally by torchvision model")
