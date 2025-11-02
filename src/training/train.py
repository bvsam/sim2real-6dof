"""
Training script for NOCS R-CNN.

Implements the 3-stage training schedule from the NOCS paper:
- Stage 1: Freeze all ResNet, train heads/RPN/FPN (10K iter, LR=0.001)
- Stage 2: Freeze below C4 (3K iter, LR=0.0001)
- Stage 3: Freeze below C3 (70K iter, LR=0.00001)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.data.dataset import NOCSDataset
from src.training.losses.nocs_losses import (
    BBoxLoss,
    ClassificationLoss,
    MaskLoss,
    NOCSLoss,
)
from src.training.model.nocs_rcnn import NOCSRCNN
from src.training.utils.training_utils import ProposalTargetMatcher

logger = logging.getLogger(__name__)


class NOCSTrainer:
    """Trainer for NOCS R-CNN with 3-stage progressive training."""

    def __init__(
        self,
        model: NOCSRCNN,
        train_dataset: NOCSDataset,
        val_dataset: NOCSDataset,
        output_dir: str,
        device: str = "cuda",
        batch_size: int = 2,
        num_workers: int = 4,
        # Loss weights
        loss_weight_cls: float = 1.0,
        loss_weight_bbox: float = 1.0,
        loss_weight_mask: float = 1.0,
        loss_weight_nocs: float = 1.0,
        # Training stages
        stage1_iterations: int = 10000,
        stage2_iterations: int = 3000,
        stage3_iterations: int = 70000,
        stage1_lr: float = 0.001,
        stage2_lr: float = 0.0001,
        stage3_lr: float = 0.00001,
        # Optimizer
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        # Checkpointing
        checkpoint_interval: int = 1000,
        validation_interval: int = 250,
        log_interval: int = 100,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.device = device
        self.batch_size = batch_size

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

        # Loss functions
        self.loss_weight_cls = loss_weight_cls
        self.loss_weight_bbox = loss_weight_bbox
        self.loss_weight_mask = loss_weight_mask
        self.loss_weight_nocs = loss_weight_nocs

        self.cls_loss_fn = ClassificationLoss()
        self.bbox_loss_fn = BBoxLoss()
        self.mask_loss_fn = MaskLoss()
        self.nocs_loss_fn = NOCSLoss(num_bins=model.num_bins)

        # Proposal matcher
        self.proposal_matcher = ProposalTargetMatcher()

        # Training stages
        self.stages = [
            {
                "name": "Stage 1",
                "iterations": stage1_iterations,
                "lr": stage1_lr,
                "freeze_stage": 5,  # Freeze all ResNet
            },
            {
                "name": "Stage 2",
                "iterations": stage2_iterations,
                "lr": stage2_lr,
                "freeze_stage": 4,  # Freeze below C4
            },
            {
                "name": "Stage 3",
                "iterations": stage3_iterations,
                "lr": stage3_lr,
                "freeze_stage": 3,  # Freeze below C3
            },
        ]

        # Optimizer (will be reset for each stage)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer = None

        # Tracking
        self.current_stage = 0
        self.current_iteration = 0
        self.checkpoint_interval = checkpoint_interval
        self.validation_interval = validation_interval
        self.log_interval = log_interval

        # Tensorboard
        self.writer = SummaryWriter(self.output_dir / "tensorboard")

        logger.info(f"Trainer initialized:")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Validation samples: {len(val_dataset)}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Device: {device}")

    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        # Batch images
        images = torch.stack(
            [torch.from_numpy(item["image"]).permute(2, 0, 1) for item in batch]
        )
        images = images.float() / 255.0  # Normalize to [0, 1]

        # Collect targets (lists, not batched due to variable sizes)
        gt_boxes = []
        gt_labels = []
        gt_masks = []
        gt_nocs = []

        for item in batch:
            # Boxes from masks (since we have instance masks)
            mask = item["masks"][:, :, 0]  # [H, W, 1] -> [H, W]
            if mask.sum() > 0:
                # Get bounding box from mask
                ys, xs = torch.from_numpy(mask).nonzero(as_tuple=True)
                x1, y1 = xs.min().item(), ys.min().item()
                x2, y2 = xs.max().item(), ys.max().item()
                gt_boxes.append(torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32))
            else:
                # Empty image (negative sample)
                gt_boxes.append(torch.zeros((0, 4), dtype=torch.float32))

            # Labels
            gt_labels.append(torch.from_numpy(item["class_ids"]).long())

            # Masks (remove instance dimension for single-instance case)
            # [H, W, 1] -> [1, H, W]
            if item["masks"].shape[-1] > 0:
                gt_masks.append(torch.from_numpy(item["masks"][:, :, 0]).unsqueeze(0))
            else:
                h, w = item["masks"].shape[:2]
                gt_masks.append(torch.zeros((0, h, w)))

            # NOCS coords [H, W, 1, 3] -> [1, 3, H, W]
            if item["coords"].shape[2] > 0:
                coords = torch.from_numpy(item["coords"][:, :, 0, :])  # [H, W, 3]
                coords = coords.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                gt_nocs.append(coords)
            else:
                h, w = item["coords"].shape[:2]
                gt_nocs.append(torch.zeros((0, 3, h, w)))

        return {
            "images": images,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "gt_masks": gt_masks,
            "gt_nocs": gt_nocs,
        }

    def _setup_stage(self, stage_idx: int):
        """Setup model and optimizer for a training stage."""
        stage = self.stages[stage_idx]

        logger.info("=" * 70)
        logger.info(f"Setting up {stage['name']}")
        logger.info(f"  Iterations: {stage['iterations']}")
        logger.info(f"  Learning rate: {stage['lr']}")
        logger.info(f"  Freeze stage: {stage['freeze_stage']}")
        logger.info("=" * 70)

        # Freeze backbone stages
        self.model.freeze_backbone_stages(stage["freeze_stage"])

        # Create optimizer
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=stage["lr"],
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"  Trainable parameters: {trainable_params:,} / {total_params:,}")

    def _compute_losses(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}

        # Classification loss
        cls_loss = self.cls_loss_fn(outputs["class_logits"], targets["labels"])
        losses["cls"] = cls_loss * self.loss_weight_cls

        # BBox regression loss
        bbox_loss = self.bbox_loss_fn(
            outputs["bbox_deltas"],
            targets["bbox_targets"],
            targets["labels"],
            self.model.num_classes,
        )
        losses["bbox"] = bbox_loss * self.loss_weight_bbox

        # Mask loss
        mask_loss = self.mask_loss_fn(
            outputs["mask_logits"], targets["masks"], targets["labels"]
        )
        losses["mask"] = mask_loss * self.loss_weight_mask

        # NOCS loss
        nocs_loss, nocs_loss_dict = self.nocs_loss_fn(
            outputs["nocs_logits"],
            targets["nocs_coords"],
            targets["masks"],
            targets["labels"],
        )
        losses["nocs"] = nocs_loss * self.loss_weight_nocs
        losses.update({f"nocs_{k}": v for k, v in nocs_loss_dict.items()})

        # Total loss
        losses["total"] = (
            losses["cls"] + losses["bbox"] + losses["mask"] + losses["nocs"]
        )

        return losses

    def train_iteration(self, batch: Dict) -> Dict[str, float]:
        """Single training iteration."""
        self.model.train()

        # Move to device
        images = batch["images"].to(self.device)
        gt_boxes = [b.to(self.device) for b in batch["gt_boxes"]]
        gt_labels = [l.to(self.device) for l in batch["gt_labels"]]
        gt_masks = [m.to(self.device) for m in batch["gt_masks"]]
        gt_nocs = [n.to(self.device) for n in batch["gt_nocs"]]

        # Forward pass to get proposals
        with torch.no_grad():
            proposals, _ = self.model.rpn(
                self.model.backbone(images),
                [(img.shape[-2], img.shape[-1]) for img in images],
            )

        # Match proposals to ground truth
        matched_proposals, targets = self.proposal_matcher(
            proposals, gt_boxes, gt_labels, gt_masks, gt_nocs
        )

        # Skip if no valid proposals
        if targets["labels"].shape[0] == 0:
            return {"total": 0.0, "cls": 0.0, "bbox": 0.0, "mask": 0.0, "nocs": 0.0}

        # Forward pass through heads
        outputs = self.model(images, proposals=matched_proposals)

        # Compute losses
        losses = self._compute_losses(outputs, targets)

        # Backward pass
        self.optimizer.zero_grad()
        losses["total"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Return loss values
        return {
            k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_losses = {
            "cls": 0.0,
            "bbox": 0.0,
            "mask": 0.0,
            "nocs": 0.0,
            "total": 0.0,
        }
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            images = batch["images"].to(self.device)
            gt_boxes = [b.to(self.device) for b in batch["gt_boxes"]]
            gt_labels = [l.to(self.device) for l in batch["gt_labels"]]
            gt_masks = [m.to(self.device) for m in batch["gt_masks"]]
            gt_nocs = [n.to(self.device) for n in batch["gt_nocs"]]

            # Get proposals
            proposals, _ = self.model.rpn(
                self.model.backbone(images),
                [(img.shape[-2], img.shape[-1]) for img in images],
            )

            # Match proposals
            matched_proposals, targets = self.proposal_matcher(
                proposals, gt_boxes, gt_labels, gt_masks, gt_nocs
            )

            if targets["labels"].shape[0] == 0:
                continue

            # Forward pass
            outputs = self.model(images, proposals=matched_proposals)

            # Compute losses
            losses = self._compute_losses(outputs, targets)

            for k in total_losses.keys():
                if k in losses:
                    total_losses[k] += (
                        losses[k].item()
                        if isinstance(losses[k], torch.Tensor)
                        else losses[k]
                    )

            num_batches += 1

        # Average losses
        if num_batches > 0:
            for k in total_losses.keys():
                total_losses[k] /= num_batches

        return total_losses

    def save_checkpoint(self, iteration: int, stage: int):
        """Save training checkpoint."""
        checkpoint_path = (
            self.checkpoint_dir / f"checkpoint_stage{stage}_iter{iteration}.pth"
        )

        torch.save(
            {
                "iteration": iteration,
                "stage": stage,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")

        iteration = 0

        for stage_idx, stage in enumerate(self.stages):
            self._setup_stage(stage_idx)

            stage_iterations = stage["iterations"]
            stage_iter = 0

            pbar = tqdm(total=stage_iterations, desc=stage["name"])

            while stage_iter < stage_iterations:
                for batch in self.train_loader:
                    # Train iteration
                    losses = self.train_iteration(batch)

                    iteration += 1
                    stage_iter += 1

                    # Logging
                    if iteration % self.log_interval == 0:
                        log_str = f"[Iter {iteration}] "
                        log_str += " | ".join(
                            [f"{k}: {v:.4f}" for k, v in losses.items()]
                        )
                        pbar.set_postfix_str(log_str)

                        # Tensorboard
                        for k, v in losses.items():
                            self.writer.add_scalar(f"train/{k}", v, iteration)

                    # Validation (now separate from checkpointing)
                    if iteration % self.validation_interval == 0:
                        val_losses = self.validate()
                        val_str = " | ".join(
                            [f"{k}: {v:.4f}" for k, v in val_losses.items()]
                        )
                        logger.info(f"[Iter {iteration}] Validation: {val_str}")

                        for k, v in val_losses.items():
                            self.writer.add_scalar(f"val/{k}", v, iteration)

                        # Set back to training mode
                        self.model.train()

                    # Checkpointing
                    if iteration % self.checkpoint_interval == 0:
                        self.save_checkpoint(iteration, stage_idx)

                    pbar.update(1)

                    if stage_iter >= stage_iterations:
                        break

            pbar.close()

            # Save at end of stage
            self.save_checkpoint(iteration, stage_idx)

        logger.info("Training complete!")
        self.writer.close()


def setup_logging(log_file=None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] [%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def main():
    parser = argparse.ArgumentParser(description="Train NOCS R-CNN")

    # Data
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data directory (HDF5 files)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        required=True,
        help="Path to validation data directory (HDF5 files)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/training",
        help="Output directory for checkpoints and logs",
    )

    # Model
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes (including background)",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=32,
        help="Number of bins for NOCS discretization",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained ResNet backbone",
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )

    # Loss weights
    parser.add_argument("--loss-weight-cls", type=float, default=1.0)
    parser.add_argument("--loss-weight-bbox", type=float, default=1.0)
    parser.add_argument("--loss-weight-mask", type=float, default=1.0)
    parser.add_argument("--loss-weight-nocs", type=float, default=1.0)

    # Training stages (from NOCS paper)
    parser.add_argument("--stage1-iter", type=int, default=10000)
    parser.add_argument("--stage2-iter", type=int, default=3000)
    parser.add_argument("--stage3-iter", type=int, default=70000)
    parser.add_argument("--stage1-lr", type=float, default=0.001)
    parser.add_argument("--stage2-lr", type=float, default=0.0001)
    parser.add_argument("--stage3-lr", type=float, default=0.00001)

    # Checkpointing
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=100)

    args = parser.parse_args()

    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "training.log")

    logger.info("=" * 70)
    logger.info("NOCS R-CNN Training")
    logger.info("=" * 70)
    logger.info(f"Arguments:\n{json.dumps(vars(args), indent=2)}")

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = NOCSDataset(
        data_dir=args.train_data,
        split="train",
        include_negatives=False,
        load_poses=False,
    )

    val_dataset = NOCSDataset(
        data_dir=args.val_data,
        split="val",
        include_negatives=False,
        load_poses=False,
    )

    # Create model
    logger.info("Creating model...")
    model = NOCSRCNN(
        num_classes=args.num_classes,
        num_bins=args.num_bins,
        pretrained_backbone=args.pretrained,
    )

    # Create trainer
    trainer = NOCSTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        loss_weight_cls=args.loss_weight_cls,
        loss_weight_bbox=args.loss_weight_bbox,
        loss_weight_mask=args.loss_weight_mask,
        loss_weight_nocs=args.loss_weight_nocs,
        stage1_iterations=args.stage1_iter,
        stage2_iterations=args.stage2_iter,
        stage3_iterations=args.stage3_iter,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        stage3_lr=args.stage3_lr,
        checkpoint_interval=args.checkpoint_interval,
        validation_interval=args.validation_interval,
        log_interval=args.log_interval,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
