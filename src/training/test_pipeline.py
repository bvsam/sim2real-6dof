"""
Test script to verify the training pipeline works end-to-end.
"""

import logging
import sys
from pathlib import Path

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from configs.nocs_config import add_nocs_config
from data.dataset_mapper import NOCSDatasetMapper
from data.register_dataset import register_nocs_datasets
from models.nocs_rcnn import NOCSROIHeads

setup_logger()
logger = logging.getLogger("detectron2")


def test_dataset_loading(data_dir):
    """Test that dataset loads correctly."""
    logger.info("=" * 70)
    logger.info("TEST 1: Dataset Loading")
    logger.info("=" * 70)

    from data.dataset import NOCSDataset

    dataset = NOCSDataset(
        data_dir=data_dir,
        split="train",
        include_negatives=False,
        load_poses=True,
    )

    logger.info(f"✓ Dataset loaded: {len(dataset)} samples")

    # Load one sample
    sample = dataset[0]
    logger.info(f"✓ Sample 0 loaded:")
    logger.info(f"  - Image shape: {sample['image'].shape}")
    logger.info(f"  - Masks shape: {sample['masks'].shape}")
    logger.info(f"  - Coords shape: {sample['coords'].shape}")
    logger.info(f"  - Class IDs: {sample['class_ids']}")
    logger.info(f"  - Model: {sample['model_name']}")

    # Check class ID is 0-indexed
    assert (
        sample["class_ids"][0] == 0
    ), f"Class ID should be 0, got {sample['class_ids'][0]}"
    logger.info(f"✓ Class ID is correctly 0-indexed")

    return True


def test_dataset_mapper(cfg):
    """Test dataset mapper."""
    logger.info("=" * 70)
    logger.info("TEST 2: Dataset Mapper")
    logger.info("=" * 70)

    from data.dataset import NOCSDataset

    dataset = NOCSDataset(
        data_dir=cfg.DATASETS.TRAIN[0].replace("nocs_", ""),
        split="train",
        include_negatives=False,
    )

    # Get raw sample
    raw_sample = dataset[0]

    # Convert to dict format expected by mapper
    sample_dict = {
        "image": raw_sample["image"],
        "masks": raw_sample["masks"],
        "coords": raw_sample["coords"],
        "class_ids": raw_sample["class_ids"],
        "file_path": raw_sample["file_path"],
        "image_id": 0,
        "height": raw_sample["image"].shape[0],
        "width": raw_sample["image"].shape[1],
    }

    # Apply mapper
    mapper = NOCSDatasetMapper(cfg, is_train=True)
    mapped_sample = mapper(sample_dict)

    logger.info(f"✓ Mapper applied successfully")
    logger.info(f"  - Image tensor shape: {mapped_sample['image'].shape}")
    logger.info(f"  - Instances: {mapped_sample['instances']}")
    logger.info(f"  - gt_boxes: {mapped_sample['instances'].gt_boxes}")
    logger.info(f"  - gt_classes: {mapped_sample['instances'].gt_classes}")
    logger.info(f"  - gt_coords shape: {mapped_sample['instances'].gt_coords.shape}")
    logger.info(
        f"  - gt_nocs_masks shape: {mapped_sample['instances'].gt_nocs_masks.shape}"
    )

    # Verify shapes
    assert mapped_sample["instances"].gt_coords.shape[1:] == (28, 28, 3)
    assert mapped_sample["instances"].gt_nocs_masks.shape[1:] == (28, 28)

    logger.info(f"✓ All shapes correct")

    return True


def test_model_forward(cfg, data_dir):
    """Test model forward pass."""
    logger.info("=" * 70)
    logger.info("TEST 3: Model Forward Pass")
    logger.info("=" * 70)

    # Register dataset
    register_nocs_datasets(data_dir, splits=["train"])

    # Build model
    from detectron2.modeling import build_model
    from detectron2.utils.events import EventStorage  # ADD THIS
    from models.nocs_rcnn import NOCSROIHeads

    model = build_model(cfg)
    model.train()

    logger.info(f"✓ Model built: {model.__class__.__name__}")
    logger.info(f"  - ROI Heads: {model.roi_heads.__class__.__name__}")
    logger.info(f"  - Has NOCS head: {hasattr(model.roi_heads, 'nocs_head')}")

    # Build dataloader
    mapper = NOCSDatasetMapper(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)

    # Get one batch
    data_iter = iter(data_loader)
    batch = next(data_iter)

    logger.info(f"✓ Batch loaded: {len(batch)} samples")

    # Forward pass - WRAP IN EventStorage
    with EventStorage() as storage:  # ADD THIS
        with torch.no_grad():
            outputs = model(batch)

    logger.info(f"✓ Forward pass successful")
    logger.info(f"  - Losses: {list(outputs.keys())}")

    # Check NOCS loss exists
    assert "loss_nocs" in outputs or "loss_nocs_x" in outputs, "NOCS loss not found!"
    logger.info(f"✓ NOCS loss present")

    return True


def test_training_iteration(cfg, data_dir):
    """Test one training iteration."""
    logger.info("=" * 70)
    logger.info("TEST 4: Training Iteration")
    logger.info("=" * 70)

    # Register dataset
    register_nocs_datasets(data_dir, splits=["train"])

    # Build model
    from detectron2.modeling import build_model
    from detectron2.utils.events import EventStorage  # ADD THIS

    model = build_model(cfg)
    model.train()

    # Build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    # Build dataloader
    mapper = NOCSDatasetMapper(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    data_iter = iter(data_loader)

    # Training iteration - WRAP IN EventStorage
    batch = next(data_iter)

    optimizer.zero_grad()

    with EventStorage() as storage:  # ADD THIS
        loss_dict = model(batch)
        losses = sum(loss_dict.values())

    logger.info(f"✓ Loss computed: {losses.item():.4f}")
    logger.info(f"  - Loss breakdown:")
    for k, v in loss_dict.items():
        logger.info(f"    {k}: {v.item():.4f}")

    losses.backward()
    optimizer.step()

    logger.info(f"✓ Backward pass and optimizer step successful")

    return True


def main():
    """Run all tests."""
    # Setup
    data_dir = "/home/blender/workspace/output"  # Change this to your data directory

    # Create config - START WITH BASE CONFIG
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    cfg = get_cfg()

    # Load base Mask R-CNN config first
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    # Now add NOCS config
    add_nocs_config(cfg)

    # Override with our custom settings
    cfg.MODEL.ROI_HEADS.NAME = "NOCSROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.NOCS_HEAD.NUM_BINS = 32
    cfg.MODEL.NOCS_HEAD.CONV_DIM = 512
    cfg.MODEL.NOCS_HEAD.NUM_CONV = 8
    cfg.MODEL.NOCS_HEAD.USE_BN = True
    cfg.MODEL.NOCS_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.NOCS_HEAD.LOSS_WEIGHT = 1.0

    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.FLIP_Z_AXIS = True

    cfg.DATASETS.TRAIN = (f"nocs_train",)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.DATALOADER.NUM_WORKERS = 0  # For testing

    # Use CPU for testing
    cfg.MODEL.DEVICE = "cpu"

    # Reduce image size for faster testing
    cfg.INPUT.MIN_SIZE_TRAIN = (480,)
    cfg.INPUT.MAX_SIZE_TRAIN = 640

    logger.info(f"Using device: {cfg.MODEL.DEVICE}")

    # Run tests
    try:
        test_dataset_loading(data_dir)
        test_dataset_mapper(cfg)
        test_model_forward(cfg, data_dir)
        test_training_iteration(cfg, data_dir)

        logger.info("=" * 70)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 70)
        logger.info("Your pipeline is ready for training!")
        logger.info(
            "Run: python train.py --config-file configs/nocs_mrcnn_R50_FPN.yaml --data-dir /path/to/data"
        )

    except Exception as e:
        logger.error("=" * 70)
        logger.error("❌ TEST FAILED!")
        logger.error("=" * 70)
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
    sys.exit(0 if success else 1)
    sys.exit(0 if success else 1)
