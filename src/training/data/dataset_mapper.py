"""
Dataset mapper to convert HDF5 samples to Detectron2 format.
"""

import copy
import json
import logging
from typing import Dict, List

import cv2
import h5py
import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Boxes, BoxMode, Instances

logger = logging.getLogger(__name__)


class NOCSDatasetMapper:
    """
    Mapper that converts HDF5 samples to Detectron2's standard format.

    Handles:
    - Loading RGB images, masks, NOCS maps from HDF5
    - Applying data augmentations
    - Formatting as Detectron2 Instances
    """

    def __init__(self, cfg, is_train=True):
        """
        Args:
            cfg: Detectron2 config
            is_train: Whether in training mode (enables augmentations)
        """
        self.is_train = is_train
        self.flip_z_axis = cfg.INPUT.FLIP_Z_AXIS  # Match NOCS convention
        self.nocs_resolution = 28  # Target resolution for NOCS maps

        # Build augmentations
        self.augmentations = self._build_augmentations(cfg, is_train)

        # Image format
        self.image_format = cfg.INPUT.FORMAT  # "BGR" or "RGB"

        logger.info(f"Dataset mapper initialized (is_train={is_train})")

    def _build_augmentations(self, cfg, is_train):
        """Build augmentation pipeline."""
        augmentations = []

        if is_train:
            # Random flip
            augmentations.append(
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False)
            )

            # TODO: Add more augmentations if needed
            # - Color jittering (brightness, contrast, saturation)
            # - Random crop/resize
            # - Rotation

        # Resize to fixed size (optional, for batching)
        # augmentations.append(T.Resize((cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN)))

        return augmentations

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict: Dict with keys from our HDF5 dataset
                - file_path: path to HDF5 file
                - image: RGB image
                - masks: instance masks
                - coords: NOCS coordinates
                - ... (other fields)

        Returns:
            Dict in Detectron2 format with "image" and "instances"
        """
        dataset_dict = copy.deepcopy(dataset_dict)

        # Load image
        image = dataset_dict["image"]  # Already loaded as numpy array [H, W, 3]

        # Detectron2 expects BGR by default
        if self.image_format == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Convert to float for augmentations
        image = image.astype(np.float32)

        # Apply augmentations
        aug_input = T.AugInput(image)
        transforms = aug_input.apply_augmentations(self.augmentations)
        image = aug_input.image

        # Convert to tensor
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["image"] = image

        # Process annotations (only during training or if available)
        if not self.is_train or "masks" not in dataset_dict:
            # Remove annotation fields for inference
            dataset_dict.pop("masks", None)
            dataset_dict.pop("coords", None)
            dataset_dict.pop("class_ids", None)
            return dataset_dict

        # Load annotations
        masks = dataset_dict["masks"]  # [H, W, num_instances]
        coords = dataset_dict["coords"]  # [H, W, num_instances, 3]
        class_ids = dataset_dict["class_ids"]  # [num_instances]

        num_instances = masks.shape[2] if len(masks.shape) == 3 else 0

        if num_instances == 0:
            # No instances (negative sample or background)
            dataset_dict["instances"] = Instances(image.shape[-2:])
            return dataset_dict

        # Apply same transforms to masks and coords
        masks_list = []
        coords_list = []

        for i in range(num_instances):
            mask = masks[:, :, i]
            coord = coords[:, :, i, :]

            # Apply transforms
            mask_transformed = transforms.apply_segmentation(mask)

            # For coords, we need to apply the same spatial transforms
            # Treat each channel separately
            coord_transformed = np.stack(
                [transforms.apply_image(coord[:, :, c]) for c in range(3)], axis=2
            )

            masks_list.append(mask_transformed)
            coords_list.append(coord_transformed)

        # Resize masks and coords to standard resolution (28x28)
        masks_resized = []
        coords_resized = []

        for mask, coord in zip(masks_list, coords_list):
            # Resize mask
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (self.nocs_resolution, self.nocs_resolution),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            # Resize coords
            coord_resized = cv2.resize(
                coord,
                (self.nocs_resolution, self.nocs_resolution),
                interpolation=cv2.INTER_LINEAR,
            )

            # Flip Z-axis if needed (to match NOCS convention)
            if self.flip_z_axis:
                coord_resized[:, :, 2] = 1.0 - coord_resized[:, :, 2]

            masks_resized.append(mask_resized)
            coords_resized.append(coord_resized)

        # Stack into tensors
        masks_tensor = torch.stack(
            [torch.from_numpy(m.copy()) for m in masks_resized]
        )  # [N, 28, 28]
        coords_tensor = torch.stack(
            [torch.from_numpy(c.copy()) for c in coords_resized]
        )  # [N, 28, 28, 3]

        # Create bounding boxes from masks
        boxes = []
        for mask in masks_list:
            # Get bounding box from full-resolution mask
            pos = np.where(mask)
            if len(pos[0]) == 0:
                # Empty mask, use dummy box
                boxes.append([0, 0, 1, 1])
            else:
                y_min, y_max = pos[0].min(), pos[0].max()
                x_min, x_max = pos[1].min(), pos[1].max()
                boxes.append([x_min, y_min, x_max, y_max])

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)

        # Create Detectron2 Instances
        instances = Instances(image.shape[-2:])
        instances.gt_boxes = Boxes(boxes_tensor)
        instances.gt_classes = torch.as_tensor(class_ids, dtype=torch.int64)
        instances.gt_masks = BitMasks(
            torch.stack([torch.from_numpy(m.copy()) for m in masks_list])
        )
        instances.gt_coords = coords_tensor
        instances.gt_nocs_masks = masks_tensor  # Resized masks for NOCS loss

        dataset_dict["instances"] = instances

        return dataset_dict
