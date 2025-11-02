"""
Custom evaluator for NOCS model.

Computes metrics from the NOCS paper:
- 3D IoU AP (Average Precision at 50% IoU)
- Rotation AP (at various degree thresholds)
- Translation AP (at various cm thresholds)
"""

import copy
import logging
import os
import pickle
from collections import OrderedDict, defaultdict
from typing import Dict, List

import numpy as np
import torch
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm

from .pose_estimator import (
    compute_3d_iou,
    compute_rotation_error,
    compute_translation_error,
    estimate_pose_from_nocs,
)

logger = logging.getLogger(__name__)


class NOCSEvaluator(DatasetEvaluator):
    """
    Evaluate NOCS predictions and 6DoF pose estimation.

    Metrics:
    - 3D Detection: IoU@50
    - Rotation: AP at 5°, 10°, 15° thresholds
    - Translation: AP at 2cm, 5cm, 10cm thresholds
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name: Name of the dataset
            cfg: Detectron2 config
            distributed: Whether running in distributed mode
            output_dir: Directory to save results
        """
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")

        # Evaluation thresholds
        self.rotation_thresholds = [5, 10, 15]  # degrees
        self.translation_thresholds = [0.02, 0.05, 0.10]  # meters (2cm, 5cm, 10cm)
        self.iou_threshold = 0.5
        self.detection_threshold = 0.1  # Minimum bbox overlap for evaluation

        logger.info(f"NOCSEvaluator initialized for {dataset_name}")

    def reset(self):
        """Reset internal state."""
        self._predictions = []
        self._ground_truths = []

    def process(self, inputs, outputs):
        """
        Process one batch of predictions.

        Args:
            inputs: List of dataset dicts (ground truth)
            outputs: List of inference results from the model
        """
        for input_dict, output_dict in zip(inputs, outputs):
            prediction = {
                "image_id": input_dict["image_id"],
                "instances": output_dict["instances"].to(self._cpu_device),
            }

            ground_truth = {
                "image_id": input_dict["image_id"],
                "height": input_dict["height"],
                "width": input_dict["width"],
                "instances": input_dict.get("instances", None),
            }

            # Add GT pose if available
            if "rotation" in input_dict:
                ground_truth["rotation"] = input_dict["rotation"]
                ground_truth["translation"] = input_dict["translation"]

            # Add camera intrinsics (needed for pose estimation)
            if "intrinsics" in input_dict:
                ground_truth["intrinsics"] = input_dict["intrinsics"]

            self._predictions.append(prediction)
            self._ground_truths.append(ground_truth)

    def evaluate(self):
        """
        Evaluate accumulated predictions.

        Returns:
            OrderedDict with evaluation metrics
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = [p for sublist in predictions for p in sublist]

            ground_truths = comm.gather(self._ground_truths, dst=0)
            ground_truths = [g for sublist in ground_truths for g in sublist]

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            ground_truths = self._ground_truths

        if len(predictions) == 0:
            logger.warning("No predictions to evaluate!")
            return {}

        # Compute metrics
        results = self._evaluate_predictions(predictions, ground_truths)

        # Save detailed results
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            file_path = os.path.join(self._output_dir, "nocs_evaluation_results.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(results, f)
            logger.info(f"Saved evaluation results to {file_path}")

        return results

    def _evaluate_predictions(self, predictions, ground_truths):
        """Compute all evaluation metrics."""
        # Storage for matched predictions
        matches = []

        for pred, gt in zip(predictions, ground_truths):
            pred_instances = pred["instances"]
            gt_instances = gt.get("instances", None)

            if gt_instances is None or len(gt_instances) == 0:
                continue

            # For single-instance case (one object per image)
            # TODO: Extend to multi-instance if needed

            if len(pred_instances) == 0:
                # No detection
                matches.append(
                    {
                        "detected": False,
                        "iou_3d": 0.0,
                        "rotation_error": None,
                        "translation_error": None,
                    }
                )
                continue

            # Take highest-scoring prediction
            scores = pred_instances.scores.numpy()
            best_idx = np.argmax(scores)

            # Get predictions
            pred_mask = pred_instances.pred_masks[best_idx].numpy()  # [H, W]
            pred_coords = pred_instances.pred_coords[best_idx].numpy()  # [28, 28, 3]
            pred_score = scores[best_idx]

            # Get ground truth
            gt_mask = gt_instances.gt_masks.tensor[0].numpy()  # [H, W]
            gt_coords = gt_instances.gt_coords[0].numpy()  # [28, 28, 3]

            # Compute 3D IoU
            iou_3d = compute_3d_iou(pred_coords, gt_coords, pred_mask, gt_mask)

            # Estimate pose (if GT depth available - for synthetic data, we can synthesize)
            # For now, we'll skip pose estimation and focus on NOCS metrics
            # TODO: Add depth map synthesis or use GT depth

            match = {
                "detected": True,
                "score": pred_score,
                "iou_3d": iou_3d,
                "rotation_error": None,  # TODO
                "translation_error": None,  # TODO
            }

            matches.append(match)

        # Compute metrics
        results = OrderedDict()

        # Detection rate
        num_detected = sum(1 for m in matches if m["detected"])
        detection_rate = num_detected / len(matches) if len(matches) > 0 else 0
        results["detection_rate"] = detection_rate

        # 3D IoU AP
        ious = [m["iou_3d"] for m in matches if m["detected"]]
        if len(ious) > 0:
            results["3d_iou_mean"] = np.mean(ious)
            results["3d_iou_ap50"] = np.mean([iou > 0.5 for iou in ious])
        else:
            results["3d_iou_mean"] = 0.0
            results["3d_iou_ap50"] = 0.0

        # Rotation/Translation AP (placeholder - need depth for full implementation)
        results["rotation_ap_5deg"] = 0.0  # TODO
        results["rotation_ap_10deg"] = 0.0  # TODO
        results["translation_ap_2cm"] = 0.0  # TODO
        results["translation_ap_5cm"] = 0.0  # TODO

        # Log results
        logger.info("=" * 60)
        logger.info("NOCS Evaluation Results:")
        logger.info("=" * 60)
        for key, value in results.items():
            logger.info(f"{key}: {value:.4f}")
        logger.info("=" * 60)

        return results
