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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.data.dataset import NOCSDataset
from src.training.losses.nocs_losses import nocs_loss
from src.training.model.nocs_maskrcnn import NOCSMaskRCNN

logger = logging.getLogger(__name__)


class NOCSTrainer:
    """Trainer for NOCS R-CNN with 3-stage progressive training."""

    def __init__(
        self,
        model: NOCSMaskRCNN,
        train_dataset: NOCSDataset,
        val_dataset: NOCSDataset,
        output_dir: str,
        device: str = "cuda",
        batch_size: int = 2,
        num_workers: int = 4,
        # Loss weights
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
        # Note: torchvision models handle batching internally, so batch_size=1 for dataloader
        # and we'll manually batch inside
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
        self.loss_weight_nocs = loss_weight_nocs

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
        """
        Collate function for torchvision model format.

        Returns:
            images: List of [3, H, W] tensors
            targets: List of dicts with 'boxes', 'labels', 'masks', 'nocs'
        """
        images = []
        targets = []

        for item in batch:
            # Image: [H, W, 3] -> [3, H, W], normalized to [0, 1]
            img = torch.from_numpy(item["image"]).permute(2, 0, 1).float() / 255.0
            images.append(img)

            # Extract masks and create target dict
            mask = item["masks"][:, :, 0]  # [H, W, 1] -> [H, W]

            target = {}

            if mask.sum() > 0:
                # Bounding box from mask
                ys, xs = np.where(mask > 0)
                x1, y1 = xs.min(), ys.min()
                x2, y2 = xs.max(), ys.max()
                target["boxes"] = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)

                # Labels
                target["labels"] = torch.tensor([1], dtype=torch.int64)  # Class 1 (mug)

                # Masks: [H, W] -> [1, H, W]
                target["masks"] = torch.from_numpy(mask).unsqueeze(0).to(torch.uint8)

                # NOCS: [H, W, 1, 3] -> [1, 3, H, W]
                coords = item["coords"][:, :, 0, :]  # [H, W, 3]
                target["nocs"] = (
                    torch.from_numpy(coords).permute(2, 0, 1).unsqueeze(0).float()
                )
            else:
                # Empty sample (negative)
                h, w = mask.shape
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros((0,), dtype=torch.int64)
                target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
                target["nocs"] = torch.zeros((0, 3, h, w), dtype=torch.float32)

            targets.append(target)

        return images, targets

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

    def train_iteration(self, images, targets) -> Dict[str, float]:
        """Single training iteration."""
        self.model.train()

        # Move to device
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Forward pass (model computes internal losses)
        outputs = self.model(images, targets)
        loss_dict = outputs["losses"]

        # Compute NOCS loss externally
        nocs_logits = outputs.get("nocs_logits")
        if nocs_logits is not None and nocs_logits.shape[0] > 0:
            # Get stored data from RoI heads
            nocs_proposals = self.model.model.roi_heads.nocs_proposals_storage
            nocs_matched_idxs = self.model.model.roi_heads.nocs_matched_idxs_storage

            # Extract GT data
            gt_nocs = [t["nocs"] for t in targets]
            gt_labels = [t["labels"] for t in targets]

            # Compute NOCS loss
            loss_nocs = nocs_loss(
                nocs_logits, nocs_proposals, gt_nocs, gt_labels, nocs_matched_idxs
            )
            loss_dict["loss_nocs"] = loss_nocs * self.loss_weight_nocs
        else:
            loss_dict["loss_nocs"] = torch.tensor(0.0, device=self.device)

        # Total loss (sum all losses from dict)
        total_loss = sum(loss for loss in loss_dict.values())

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Return loss values
        loss_dict["total"] = total_loss
        return {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in loss_dict.items()
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_losses = {}
        num_batches = 0

        for images, targets in tqdm(self.val_loader, desc="Validation", leave=False):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Forward pass
            self.model.train()  # Need train mode to get losses
            outputs = self.model(images, targets)
            loss_dict = outputs["losses"]

            # Compute NOCS loss
            nocs_logits = outputs.get("nocs_logits")
            if nocs_logits is not None and nocs_logits.shape[0] > 0:
                nocs_proposals = self.model.model.roi_heads.nocs_proposals_storage
                nocs_matched_idxs = self.model.model.roi_heads.nocs_matched_idxs_storage

                gt_nocs = [t["nocs"] for t in targets]
                gt_labels = [t["labels"] for t in targets]

                loss_nocs = nocs_loss(
                    nocs_logits, nocs_proposals, gt_nocs, gt_labels, nocs_matched_idxs
                )
                loss_dict["loss_nocs"] = loss_nocs * self.loss_weight_nocs
            else:
                loss_dict["loss_nocs"] = torch.tensor(0.0, device=self.device)

            # Accumulate
            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += val

            num_batches += 1

        # Average
        if num_batches > 0:
            for k in total_losses.keys():
                total_losses[k] /= num_batches

        # Add total
        total_losses["total"] = sum(total_losses.values())

        self.model.eval()
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
                for images, targets in self.train_loader:
                    # Train iteration
                    losses = self.train_iteration(images, targets)

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

                    # Validation
                    if iteration % self.validation_interval == 0:
                        val_losses = self.validate()
                        val_str = " | ".join(
                            [f"{k}: {v:.4f}" for k, v in val_losses.items()]
                        )
                        logger.info(f"[Iter {iteration}] Validation: {val_str}")

                        for k, v in val_losses.items():
                            self.writer.add_scalar(f"val/{k}", v, iteration)

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
    model = NOCSMaskRCNN(
        num_classes=args.num_classes,
        num_bins=args.num_bins,
        pretrained=args.pretrained,
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
