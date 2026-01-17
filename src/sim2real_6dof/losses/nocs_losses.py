"""
Loss functions for NOCS R-CNN training.

Key insight from the paper: NOCS coordinates are predicted via
CLASSIFICATION (binning), not regression.
"""

import torch
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
    batch_inds = torch.zeros_like(matched_idxs[:, None])
    rois = torch.cat([batch_inds, boxes], dim=1)
    gt_nocs = gt_nocs.to(rois)
    return roi_align(gt_nocs, rois, (M, M), 1.0, aligned=True)


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    """
    Project binary masks onto proposal boxes.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks.to(rois).float()
    if gt_masks.dim() == 3:
        gt_masks = gt_masks.unsqueeze(1)
    # Use nearest neighbor (or bilnear > 0.5) for masks
    return roi_align(gt_masks, rois, (M, M), 1.0, aligned=True)


def nocs_loss(nocs_logits, proposals, gt_nocs, gt_masks, gt_labels, nocs_matched_idxs):
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

    # Resize GT NOCS to match the spatial dimensions of GT Masks (which MaskRCNN resized)
    resized_gt_nocs = []
    for nocs, masks in zip(gt_nocs, gt_masks):
        target_h, target_w = masks.shape[-2:]

        if nocs.shape[-2:] != (target_h, target_w):
            nocs = F.interpolate(
                nocs, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        resized_gt_nocs.append(nocs)
    gt_nocs = resized_gt_nocs

    # Project GT NOCS onto proposal boxes
    nocs_targets = [
        project_nocs_on_boxes(nocs, props, idxs, discretization_size)
        for nocs, props, idxs in zip(gt_nocs, proposals, nocs_matched_idxs)
    ]

    # Project GT Masks
    mask_targets = [
        project_masks_on_boxes(masks, props, idxs, discretization_size)
        for masks, props, idxs in zip(gt_masks, proposals, nocs_matched_idxs)
    ]

    # Concatenate across batch
    labels = torch.cat(labels, dim=0)
    nocs_targets = torch.cat(nocs_targets, dim=0)  # [N, 3, M, M]
    mask_targets = torch.cat(mask_targets, dim=0)

    # Handle empty case
    if nocs_targets.numel() == 0:
        return 0

    # Convert continuous NOCS [0, 1] to bin indices [0, num_bins-1]
    nocs_targets_bins = (nocs_targets * num_bins).long()
    nocs_targets_bins = torch.clamp(nocs_targets_bins, 0, num_bins - 1)  # [N, 3, M, M]

    # Select predictions for correct class
    # nocs_logits: [N, num_classes, 3, num_bins, M, M]
    # We need: [N, 3, num_bins, M, M]
    class_specific_logits = nocs_logits[
        torch.arange(labels.shape[0], device=labels.device), labels
    ]  # [N, 3, num_bins, M, M]

    # Create binary mask
    valid_mask = (mask_targets > 0.5).squeeze(1)

    if valid_mask.sum() == 0:
        return 0

    # Compute cross-entropy loss for each coordinate
    losses = []
    for coord_idx in range(3):  # x, y, z
        coord_logits = class_specific_logits[:, coord_idx]  # [N, num_bins, M, M]
        coord_targets = nocs_targets_bins[:, coord_idx]  # [N, M, M]

        # Reshape for cross_entropy: [N*M*M, num_bins] and [N*M*M]
        coord_logits_flat = coord_logits.permute(0, 2, 3, 1).reshape(-1, num_bins)
        coord_targets_flat = coord_targets.reshape(-1)
        valid_mask_flat = valid_mask.reshape(-1)

        loss_coord = F.cross_entropy(
            coord_logits_flat[valid_mask_flat],
            coord_targets_flat[valid_mask_flat],
            reduction="mean",
        )
        losses.append(loss_coord)

    # Average over x, y, z
    total_loss = sum(losses) / 3.0

    return total_loss
