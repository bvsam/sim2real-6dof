"""
Configuration for NOCS training.
Extends Detectron2's default Mask R-CNN config.
"""

from detectron2.config import CfgNode as CN


def add_nocs_config(cfg):
    """
    Add config for NOCS head.
    """
    cfg.MODEL.NOCS_HEAD = CN()

    # Architecture
    cfg.MODEL.NOCS_HEAD.NUM_BINS = 32  # Discretization bins per coordinate
    cfg.MODEL.NOCS_HEAD.CONV_DIM = 512  # Channel dimension for conv layers
    cfg.MODEL.NOCS_HEAD.NUM_CONV = 8  # Number of conv layers at 14x14
    cfg.MODEL.NOCS_HEAD.USE_BN = True  # Use batch normalization
    cfg.MODEL.NOCS_HEAD.POOLER_RESOLUTION = 14  # ROI pooling resolution
    cfg.MODEL.NOCS_HEAD.POOLER_SAMPLING_RATIO = 0  # 0 = adaptive
    cfg.MODEL.NOCS_HEAD.POOLER_TYPE = "ROIAlignV2"

    # Loss weights
    cfg.MODEL.NOCS_HEAD.LOSS_WEIGHT = 1.0

    # Input settings (ADD THIS)
    cfg.INPUT.FLIP_Z_AXIS = True  # Flip Z-axis to match NOCS convention

    # Training
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000  # Save every 5k iterations
