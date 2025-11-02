"""
NOCS Coordinate Prediction Head for Detectron2.

Implements the NOCS head from "Normalized Object Coordinate Space for
Category-Level 6D Object Pose and Size Estimation" (Wang et al., CVPR 2019).

Uses classification approach: discretize [0,1] into 32 bins per coordinate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import Conv2d, ConvTranspose2d
from detectron2.utils.registry import Registry

NOCS_HEAD_REGISTRY = Registry("NOCS_HEAD")


@NOCS_HEAD_REGISTRY.register()
class NOCSHead(nn.Module):
    """
    NOCS prediction head using classification approach.

    Predicts 3 coordinates (x, y, z) each discretized into NUM_BINS classes.
    Architecture matches the NOCS paper:
    - 8x Conv3x3 layers at 14x14 resolution
    - 1x ConvTranspose2x2 to upsample to 28x28
    - 1x Conv1x1 to produce final predictions
    """

    def __init__(self, cfg, input_shape):
        """
        Args:
            cfg: Detectron2 config
            input_shape: ShapeSpec with channels, height, width
        """
        super().__init__()

        # Hyperparameters
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.num_bins = cfg.MODEL.NOCS_HEAD.NUM_BINS  # 32 in paper
        self.conv_dims = cfg.MODEL.NOCS_HEAD.CONV_DIM  # 512 in paper
        self.num_conv = cfg.MODEL.NOCS_HEAD.NUM_CONV  # 8 in paper
        self.use_bn = cfg.MODEL.NOCS_HEAD.USE_BN  # Batch norm

        # Build conv layers (8 layers at 14x14)
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if self.use_bn else None

        in_channels = input_shape.channels  # 256 from FPN

        for i in range(self.num_conv):
            # Use plain Conv2d instead of Detectron2's wrapper to avoid issues
            conv = nn.Conv2d(
                in_channels,
                self.conv_dims,
                kernel_size=3,
                padding=1,
                bias=not self.use_bn,
            )
            self.conv_layers.append(conv)

            if self.use_bn:
                self.bn_layers.append(nn.BatchNorm2d(self.conv_dims))

            in_channels = self.conv_dims

        # Upsampling layer (14x14 -> 28x28)
        self.deconv = nn.ConvTranspose2d(
            self.conv_dims, self.conv_dims, kernel_size=2, stride=2, padding=0
        )

        # Final prediction layer: 3 coordinates × num_bins × num_classes
        self.predictor = nn.Conv2d(
            self.conv_dims,
            self.num_classes * 3 * self.num_bins,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Initialize weights
        for conv in self.conv_layers:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)

        if self.use_bn:
            for bn in self.bn_layers:
                nn.init.constant_(bn.weight, 1)
                nn.init.constant_(bn.bias, 0)

        nn.init.kaiming_normal_(self.deconv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.deconv.bias, 0)

        nn.init.normal_(self.predictor.weight, std=0.001)
        nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Features from ROI pooling, shape [N, C, 14, 14]

        Returns:
            nocs_pred: NOCS predictions, shape [N, num_classes, 3, num_bins, 28, 28]
        """
        # Apply conv layers with ReLU
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = F.relu(x)

        # Upsample to 28x28
        x = F.relu(self.deconv(x))

        # Predict
        x = self.predictor(x)  # [N, num_classes*3*num_bins, 28, 28]

        # Reshape to separate classes, coordinates, and bins
        N = x.shape[0]
        x = x.view(N, self.num_classes, 3, self.num_bins, 28, 28)

        return x


def nocs_loss(pred_nocs, gt_coords, gt_masks, gt_classes, num_bins=32):
    """
    Compute NOCS loss using classification approach.

    Args:
        pred_nocs: Predicted NOCS logits [N, num_classes, 3, num_bins, 28, 28]
        gt_coords: Ground truth NOCS coordinates [N, 28, 28, 3], values in [0, 1]
        gt_masks: Instance masks [N, 28, 28], binary
        gt_classes: Class IDs [N], integers
        num_bins: Number of discretization bins (32)

    Returns:
        loss: Scalar loss value
        loss_dict: Dictionary with per-coordinate losses for logging
    """
    device = pred_nocs.device
    N = pred_nocs.shape[0]

    if N == 0:
        return pred_nocs.sum() * 0, {}

    # Convert continuous coordinates [0, 1] to bin indices [0, num_bins-1]
    # Clamp to handle edge cases
    gt_coords = torch.clamp(gt_coords, 0.0, 0.9999)
    gt_bins = (gt_coords * num_bins).long()  # [N, 28, 28, 3]

    # Gather predictions for the correct class
    # pred_nocs: [N, num_classes, 3, num_bins, 28, 28]
    batch_idx = torch.arange(N, device=device)
    pred_class_nocs = pred_nocs[batch_idx, gt_classes]  # [N, 3, num_bins, 28, 28]

    # Compute cross-entropy loss for each coordinate separately
    losses = {}
    total_loss = 0.0

    coord_names = ["x", "y", "z"]

    for coord_idx, coord_name in enumerate(coord_names):
        # Get predictions and targets for this coordinate
        coord_pred = pred_class_nocs[:, coord_idx]  # [N, num_bins, 28, 28]
        coord_target = gt_bins[:, :, :, coord_idx]  # [N, 28, 28]

        # Reshape for cross_entropy: [N*28*28, num_bins] and [N*28*28]
        coord_pred_flat = coord_pred.permute(0, 2, 3, 1).reshape(-1, num_bins)
        coord_target_flat = coord_target.reshape(-1)
        mask_flat = gt_masks.reshape(-1)

        # Apply mask: only compute loss on object pixels
        if mask_flat.sum() > 0:
            coord_pred_masked = coord_pred_flat[mask_flat > 0]
            coord_target_masked = coord_target_flat[mask_flat > 0]

            loss = F.cross_entropy(coord_pred_masked, coord_target_masked)
        else:
            loss = coord_pred_flat.sum() * 0  # Zero loss if no pixels

        losses[f"loss_nocs_{coord_name}"] = loss
        total_loss += loss

    # Average across coordinates
    losses["loss_nocs"] = total_loss / 3.0

    return losses["loss_nocs"], losses


def nocs_inference(pred_nocs, pred_classes):
    """
    Convert NOCS predictions from classification bins to continuous coordinates.

    Args:
        pred_nocs: NOCS logits [N, num_classes, 3, num_bins, 28, 28]
        pred_classes: Predicted class IDs [N]

    Returns:
        coords: NOCS coordinates [N, 28, 28, 3], values in [0, 1]
    """
    device = pred_nocs.device
    N = pred_nocs.shape[0]
    num_bins = pred_nocs.shape[3]

    if N == 0:
        return torch.zeros((0, 28, 28, 3), device=device)

    # Get predictions for the predicted class
    batch_idx = torch.arange(N, device=device)
    pred_class_nocs = pred_nocs[batch_idx, pred_classes]  # [N, 3, num_bins, 28, 28]

    # Get bin predictions (argmax over bins)
    pred_bins = pred_class_nocs.argmax(dim=2)  # [N, 3, 28, 28]

    # Convert bins to continuous coordinates
    # Map bin centers: bin i -> (i + 0.5) / num_bins
    coords = (pred_bins.float() + 0.5) / num_bins

    # Rearrange to [N, 28, 28, 3]
    coords = coords.permute(0, 2, 3, 1)

    return coords
