"""
Loss functions for NOCS R-CNN training.

Key insight from the paper: NOCS coordinates are predicted via
CLASSIFICATION (binning), not regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_nocs_bin_targets(nocs_coords_continuous, num_bins=32):
    """
    Convert continuous NOCS coordinates [0, 1] to bin indices.

    Args:
        nocs_coords_continuous: [N, 3, H, W] or [N, H, W, 3] in range [0, 1]
        num_bins: Number of bins (32 in NOCS paper)

    Returns:
        bin_indices: [N, 3, H, W] long tensor with values in [0, num_bins-1]
    """
    # Ensure values are in [0, 1]
    coords = torch.clamp(nocs_coords_continuous, 0.0, 1.0)

    # Convert to bin indices [0, num_bins-1]
    # 0.0 -> bin 0, 1.0 -> bin 31 (for 32 bins)
    bin_indices = (coords * num_bins).long()
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

    return bin_indices


class NOCSLoss(nn.Module):
    """
    NOCS coordinate loss using bin classification.

    Computes cross-entropy loss for each coordinate (x, y, z) independently.
    Loss is only computed on pixels within the object mask.
    """

    def __init__(self, num_bins=32):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, nocs_logits, target_coords, target_masks, class_ids):
        """
        Args:
            nocs_logits: [N, num_classes, 3, num_bins, H, W] predicted logits
            target_coords: [N, 3, H, W] ground truth NOCS in [0, 1]
            target_masks: [N, H, W] binary masks indicating valid pixels
            class_ids: [N] class ID for each RoI

        Returns:
            loss: Scalar tensor
            loss_dict: Dictionary with per-coordinate losses
        """
        N = nocs_logits.shape[0]
        H, W = nocs_logits.shape[-2:]

        if N == 0:
            return torch.tensor(0.0, device=nocs_logits.device), {}

        # Convert continuous coords to bin targets
        target_bins = compute_nocs_bin_targets(
            target_coords, self.num_bins
        )  # [N, 3, H, W]

        # Select logits for the correct class
        # nocs_logits: [N, num_classes, 3, num_bins, H, W]
        # We need: [N, 3, num_bins, H, W]
        class_specific_logits = torch.stack(
            [nocs_logits[i, class_ids[i]] for i in range(N)]  # [3, num_bins, H, W]
        )  # [N, 3, num_bins, H, W]

        # Compute loss for each coordinate separately
        losses = []
        loss_dict = {}

        for coord_idx, coord_name in enumerate(["x", "y", "z"]):
            # Get logits and targets for this coordinate
            coord_logits = class_specific_logits[:, coord_idx]  # [N, num_bins, H, W]
            coord_targets = target_bins[:, coord_idx]  # [N, H, W]

            # Reshape for cross-entropy: [N*H*W, num_bins] and [N*H*W]
            coord_logits = coord_logits.permute(0, 2, 3, 1).reshape(-1, self.num_bins)
            coord_targets = coord_targets.reshape(-1)
            target_masks_flat = target_masks.reshape(-1)

            # Only compute loss on valid (masked) pixels
            valid_pixels = target_masks_flat > 0

            if valid_pixels.sum() == 0:
                loss_coord = torch.tensor(0.0, device=nocs_logits.device)
            else:
                loss_coord = F.cross_entropy(
                    coord_logits[valid_pixels],
                    coord_targets[valid_pixels],
                    reduction="mean",
                )

            losses.append(loss_coord)
            loss_dict[f"nocs_{coord_name}"] = loss_coord.item()

        # Average over x, y, z
        total_loss = sum(losses) / 3.0
        loss_dict["nocs_total"] = total_loss.item()

        return total_loss, loss_dict


class MaskLoss(nn.Module):
    """Binary cross-entropy loss for mask prediction."""

    def __init__(self):
        super().__init__()

    def forward(self, mask_logits, target_masks, class_ids):
        """
        Args:
            mask_logits: [N, num_classes, H, W]
            target_masks: [N, H, W] binary masks
            class_ids: [N] class ID for each RoI

        Returns:
            loss: Scalar tensor
        """
        N = mask_logits.shape[0]

        if N == 0:
            return torch.tensor(0.0, device=mask_logits.device)

        # Select mask for the correct class
        class_specific_masks = torch.stack(
            [mask_logits[i, class_ids[i]] for i in range(N)]
        )  # [N, H, W]

        # Binary cross-entropy
        loss = F.binary_cross_entropy_with_logits(
            class_specific_masks, target_masks.float(), reduction="mean"
        )

        return loss


class ClassificationLoss(nn.Module):
    """Cross-entropy loss for object classification."""

    def __init__(self):
        super().__init__()

    def forward(self, class_logits, target_class_ids):
        """
        Args:
            class_logits: [N, num_classes]
            target_class_ids: [N] class labels

        Returns:
            loss: Scalar tensor
        """
        if class_logits.shape[0] == 0:
            return torch.tensor(0.0, device=class_logits.device)

        loss = F.cross_entropy(class_logits, target_class_ids, reduction="mean")
        return loss


class BBoxLoss(nn.Module):
    """Smooth L1 loss for bounding box regression."""

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, bbox_deltas, target_deltas, class_ids, num_classes):
        """
        Args:
            bbox_deltas: [N, num_classes*4] predicted deltas
            target_deltas: [N, 4] target deltas
            class_ids: [N] class ID for each RoI
            num_classes: Total number of classes

        Returns:
            loss: Scalar tensor
        """
        N = bbox_deltas.shape[0]

        if N == 0:
            return torch.tensor(0.0, device=bbox_deltas.device)

        # Reshape to [N, num_classes, 4]
        bbox_deltas = bbox_deltas.view(N, num_classes, 4)

        # Select deltas for correct class
        class_specific_deltas = torch.stack(
            [bbox_deltas[i, class_ids[i]] for i in range(N)]
        )  # [N, 4]

        # Smooth L1 loss
        loss = F.smooth_l1_loss(
            class_specific_deltas, target_deltas, beta=self.beta, reduction="mean"
        )

        return loss


if __name__ == "__main__":
    print("Testing Loss Functions...")

    # Test parameters
    num_rois = 5
    num_classes = 2
    num_bins = 32
    H, W = 28, 28

    # Dummy data
    nocs_logits = torch.randn(num_rois, num_classes, 3, num_bins, H, W)
    target_coords = torch.rand(num_rois, 3, H, W)  # [0, 1] range
    target_masks = torch.randint(0, 2, (num_rois, H, W)).float()
    class_ids = torch.ones(num_rois, dtype=torch.long)

    # Test NOCS loss
    print("\n1. NOCS Loss (Bin Classification):")
    nocs_loss_fn = NOCSLoss(num_bins=32)
    nocs_loss, nocs_loss_dict = nocs_loss_fn(
        nocs_logits, target_coords, target_masks, class_ids
    )
    print(f"   Total loss: {nocs_loss.item():.4f}")
    print(f"   Per-coordinate: {nocs_loss_dict}")
    assert nocs_loss.item() >= 0

    # Test Mask loss
    print("\n2. Mask Loss:")
    mask_logits = torch.randn(num_rois, num_classes, H, W)
    mask_loss_fn = MaskLoss()
    mask_loss = mask_loss_fn(mask_logits, target_masks, class_ids)
    print(f"   Loss: {mask_loss.item():.4f}")
    assert mask_loss.item() >= 0

    # Test Classification loss
    print("\n3. Classification Loss:")
    class_logits = torch.randn(num_rois, num_classes)
    cls_loss_fn = ClassificationLoss()
    cls_loss = cls_loss_fn(class_logits, class_ids)
    print(f"   Loss: {cls_loss.item():.4f}")
    assert cls_loss.item() >= 0

    # Test BBox loss
    print("\n4. BBox Loss:")
    bbox_deltas = torch.randn(num_rois, num_classes * 4)
    target_deltas = torch.randn(num_rois, 4)
    bbox_loss_fn = BBoxLoss()
    bbox_loss = bbox_loss_fn(bbox_deltas, target_deltas, class_ids, num_classes)
    print(f"   Loss: {bbox_loss.item():.4f}")
    assert bbox_loss.item() >= 0

    # Test bin target conversion
    print("\n5. NOCS Bin Conversion:")
    test_coords = torch.tensor([[[0.0, 0.5, 1.0]]])  # [1, 1, 3]
    test_coords = test_coords.permute(0, 2, 1)  # [1, 3, 1]
    bins = compute_nocs_bin_targets(test_coords, num_bins=32)
    print(f"   Coords: [0.0, 0.5, 1.0]")
    print(f"   Bins: {bins.squeeze().tolist()}")
    print(f"   Expected: [0, 16, 31]")

    print("\n✓ All loss functions test passed!")
