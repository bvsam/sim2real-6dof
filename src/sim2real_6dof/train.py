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
import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from huggingface_hub import get_token
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sim2real_6dof.data.dataset import create_webdataset
from sim2real_6dof.losses.nocs_losses import nocs_loss
from sim2real_6dof.model.nocs_maskrcnn import NOCSMaskRCNN

logger = logging.getLogger(__name__)


def train_iteration(
    model, optimizer, images, targets, device, scaler=None
) -> Dict[str, float]:
    model.train()

    # Move to device
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    optimizer.zero_grad()

    use_amp = scaler is not None
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        # Forward pass (model computes internal losses)
        outputs = model(images, targets)
        loss_dict = outputs["losses"]
        # Compute NOCS loss
        nocs_logits = outputs.get("nocs_logits")
        loss_dict["loss_nocs"] = compute_nocs_loss(
            model, nocs_logits, targets, device=device
        )
        # Total loss (sum all losses from dict)
        total_loss = sum(loss for loss in loss_dict.values())

    # Backward pass
    if scaler:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

    # Return loss values
    loss_dict["total"] = total_loss
    return {
        k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()
    }


def compute_nocs_loss(model, nocs_logits, targets, device):
    if nocs_logits is not None and nocs_logits.shape[0] > 0:
        # Get stored data from RoI heads
        nocs_proposals = model.model.roi_heads.nocs_proposals_storage
        nocs_matched_idxs = model.model.roi_heads.nocs_matched_idxs_storage

        # Extract GT data
        gt_nocs = model.model.roi_heads.nocs_gt_nocs_storage
        gt_masks = model.model.roi_heads.nocs_gt_masks_storage
        gt_labels = [t["labels"] for t in targets]

        # Compute NOCS loss
        loss_nocs = nocs_loss(
            nocs_logits,
            nocs_proposals,
            gt_nocs,
            gt_masks,
            gt_labels,
            nocs_matched_idxs,
        )
        return loss_nocs
    else:
        return torch.tensor(0.0, device=device)


@torch.no_grad()
def validate(model, dataloader, device, len_dataloader=None) -> Dict[str, float]:
    total_losses = {}
    num_batches = 0

    for images, targets in tqdm(
        dataloader, total=len_dataloader, desc="Validation", leave=False
    ):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # Forward pass
        model.train()  # Need train mode to get losses
        outputs = model(images, targets)
        loss_dict = outputs["losses"]
        # Compute NOCS loss
        nocs_logits = outputs.get("nocs_logits")
        loss_dict["loss_nocs"] = compute_nocs_loss(
            model, nocs_logits, targets, device=device
        )
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

    return total_losses


def save_checkpoint(checkpoint_dir, model, optimizer, iteration, stage):
    """Save training checkpoint."""
    checkpoint_path = checkpoint_dir / f"checkpoint_stage{stage}_iter{iteration}.pth"
    torch.save(
        {
            "iteration": iteration,
            "stage": stage,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    logger.info(f"Saved checkpoint: {checkpoint_path}")


def _collate_fn(batch):
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


def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] [%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def main():
    parser = argparse.ArgumentParser(description="Train NOCS R-CNN")
    # Data
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HF dataset repo ID to use for webdataset",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache directory for streamed datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/training"),
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pretrained ResNet backbone",
    )
    # Training
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=None, help="Prefetch factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    # Training stages
    parser.add_argument(
        "--stage-epochs",
        nargs="+",
        type=int,
        required=True,
        help="Number of epochs for each training stage",
    )
    parser.add_argument(
        "--stage-lrs",
        nargs="+",
        type=float,
        required=True,
        help="Learning rates to use during each training stage",
    )
    parser.add_argument(
        "--stage-freezes",
        nargs="+",
        type=int,
        required=True,
        help="Stage number of model backbone to freeze up to for each stage",
    )
    # Checkpointing and Logging
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Checkpoint interval when training, in number of iterations",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Logging interval when training, in number of iterations",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model with torch before training",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Train with automatic mixed precision (AMP)",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    args = parser.parse_args()

    # Validation
    if not (
        len(args.stage_epochs) == len(args.stage_lrs)
        and len(args.stage_lrs) == len(args.stage_freezes)
    ):
        raise ValueError(
            f"Arguments stage_epochs, stage_lrs and stage_freezes must be of the same length but got: {args.stage_epochs}, {args.stage_lrs}, {args.stage_freezes}"
        )

    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, output_dir / "training.log")
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("NOCS R-CNN Training")
    logger.info("=" * 70)
    # PosixPath is not JSON serializable
    args.output_dir = str(args.output_dir)
    args.cache_dir = str(args.cache_dir)
    logger.info(f"Arguments:\n{json.dumps(vars(args), indent=2)}")
    args.output_dir = Path(args.output_dir) if args.output_dir != "None" else None
    args.cache_dir = Path(args.cache_dir) if args.cache_dir != "None" else None

    # Load datasets
    logger.info("Loading datasets...")
    repo_id = args.repo_id
    N_SHARDS = 80
    TRAIN_SPLIT = 0.8
    n_train_shards = int(N_SHARDS * TRAIN_SPLIT)
    train_shard_ids = [i for i in range(n_train_shards)]
    val_shard_ids = [i for i in range(n_train_shards, N_SHARDS)]
    hf_token = get_token()
    if hf_token is None:
        logger.warning(
            "No huggingface token was sourced. Rate limiting may occur. Tip: login using `hf auth login`"
        )
    train_dataset = create_webdataset(
        repo_id,
        shard_ids=train_shard_ids,
        shuffle=False,
        hf_token=hf_token,
        cache_dir=args.cache_dir,
    )
    val_dataset = create_webdataset(
        repo_id,
        shard_ids=val_shard_ids,
        shuffle=False,
        hf_token=hf_token,
        cache_dir=args.cache_dir,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        collate_fn=_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        collate_fn=_collate_fn,
        pin_memory=True,
    )

    # Create model
    logger.info("Creating model...")
    model = NOCSMaskRCNN(
        num_classes=args.num_classes,
        num_bins=args.num_bins,
        pretrained=args.pretrained,
    )
    model = model.to(args.device)
    if args.compile:
        model = torch.compile(model)

    # Setup training utils
    stages = [
        {
            "epochs": epochs,
            "lr": lr,
            "freeze_stage": freeze,
        }
        for epochs, lr, freeze in zip(
            args.stage_epochs, args.stage_lrs, args.stage_freezes
        )
    ]
    summary_writer = SummaryWriter(output_dir / "tensorboard")
    args.checkpoint_dir = output_dir / "checkpoints"
    args.checkpoint_dir.mkdir(exist_ok=True)
    logger.info("Trainer initialized:")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Device: {args.device}")
    scaler = torch.amp.GradScaler(args.device) if args.amp else None

    # Train
    iteration_count = 0
    for stage_index, stage in enumerate(stages):
        logger.info("=" * 70)
        logger.info(f"Setting up stage {stage_index + 1}")
        logger.info(f"Epochs: {stage['epochs']}")
        logger.info(f"Learning rate: {stage['lr']}")
        logger.info(f"Freeze stage: {stage['freeze_stage']}")
        logger.info("=" * 70)
        # Freeze backbone stages
        model.freeze_backbone_stages(stage["freeze_stage"])
        # Create optimizer
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=stage["lr"],
            momentum=0.9,
            weight_decay=1e-4,
        )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Trainable parameters: {trainable_params:,} / {total_params:,}")
        n_epochs = args.stage_epochs[stage_index]
        for _ in tqdm(range(n_epochs), desc=f"Stage {stage_index + 1}"):
            for images, targets in tqdm(
                train_loader,
                total=math.ceil(len(train_shard_ids) * 1000 / args.batch_size),
                desc=f"Stage {stage_index + 1} train loop",
            ):
                # Train iteration
                losses = train_iteration(
                    model, optimizer, images, targets, args.device, scaler
                )
                iteration_count += 1

                # Logging
                if iteration_count % args.log_interval == 0:
                    log_str = f"[Iter {iteration_count}] "
                    log_str += " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
                    logger.info(log_str)
                    # Tensorboard
                    for k, v in losses.items():
                        summary_writer.add_scalar(f"train/{k}", v, iteration_count)

                # Checkpointing
                if iteration_count % args.checkpoint_interval == 0:
                    save_checkpoint(
                        args.checkpoint_dir,
                        model,
                        optimizer,
                        iteration_count,
                        stage_index,
                    )

            val_losses = validate(
                model,
                val_loader,
                args.device,
                len_dataloader=math.ceil(len(val_shard_ids) * 1000 / args.batch_size),
            )
            val_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_losses.items()])
            logger.info(f"[Iter {iteration_count}] Validation: {val_str}")
            for k, v in val_losses.items():
                summary_writer.add_scalar(f"val/{k}", v, iteration_count)

            # Save at end of stage
            save_checkpoint(
                args.checkpoint_dir,
                model,
                optimizer,
                iteration_count,
                stage_index,
            )

        logger.info("Training complete!")
        summary_writer.close()


if __name__ == "__main__":
    main()
