"""
Register our HDF5 dataset with Detectron2's dataset catalog.
"""

import logging
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog

from .dataset import NOCSDataset

logger = logging.getLogger(__name__)


def load_nocs_dataset(data_dir, split, include_negatives=False):
    """
    Load NOCS dataset and convert to Detectron2 list-of-dicts format.

    Args:
        data_dir: Directory with HDF5 files
        split: 'train', 'val', or 'test'
        include_negatives: Include negative samples

    Returns:
        List of dicts, each containing one sample
    """
    dataset = NOCSDataset(
        data_dir=data_dir,
        split=split,
        include_negatives=include_negatives,
        load_poses=True,  # For validation
    )

    # Convert to list of dicts
    dataset_dicts = []
    for i in range(len(dataset)):
        sample = dataset[i]

        # Detectron2 expects certain fields
        record = {
            "file_path": sample["file_path"],
            "image_id": i,
            "height": sample["image"].shape[0],
            "width": sample["image"].shape[1],
            "image": sample["image"],
            "masks": sample["masks"],
            "coords": sample["coords"],
            "class_ids": sample["class_ids"],
            "domain_label": sample["domain_label"],
            "model_name": sample["model_name"],
        }

        # Add optional pose data
        if "rotation" in sample:
            record["rotation"] = sample["rotation"]
            record["translation"] = sample["translation"]

        dataset_dicts.append(record)

    return dataset_dicts


def register_nocs_datasets(data_dir, splits=["train", "val"]):
    """
    Register NOCS datasets with Detectron2.

    Args:
        data_dir: Directory containing HDF5 files
        splits: List of splits to register
    """
    from detectron2.data import DatasetCatalog, MetadataCatalog

    for split in splits:
        dataset_name = f"nocs_{split}"

        # Check if already registered (for testing/re-running)
        if dataset_name in DatasetCatalog:
            logger.info(f"Dataset {dataset_name} already registered, skipping")
            continue

        # Register with DatasetCatalog
        DatasetCatalog.register(
            dataset_name,
            lambda s=split: load_nocs_dataset(data_dir, s, include_negatives=False),
        )

        # Register metadata
        MetadataCatalog.get(dataset_name).set(
            thing_classes=["mug"],  # Single category
            evaluator_type="nocs",  # Custom evaluator
        )

        logger.info(f"Registered dataset: {dataset_name}")
