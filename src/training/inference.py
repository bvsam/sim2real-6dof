"""
Inference script for NOCS R-CNN.

Loads a trained model and runs inference on images.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.model.nocs_rcnn import NOCSRCNN
from src.training.utils.pose_estimation import (
    compute_rotation_error,
    compute_translation_error,
    estimate_pose_from_nocs,
)


class NOCSInference:
    """Inference wrapper for NOCS R-CNN."""

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 2,
        num_bins: int = 32,
        device: str = "cuda",
        score_threshold: float = 0.7,
    ):
        self.device = device
        self.score_threshold = score_threshold

        # Load model
        self.model = NOCSRCNN(
            num_classes=num_classes,
            num_bins=num_bins,
            pretrained_backbone=False,
        ).to(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Loaded model from {checkpoint_path}")
        print(f"  Iteration: {checkpoint.get('iteration', 'unknown')}")
        print(f"  Stage: {checkpoint.get('stage', 'unknown')}")

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        camera_intrinsics: np.ndarray,
        depth: np.ndarray = None,
    ) -> List[Dict]:
        """
        Run inference on a single image.

        Args:
            image: [H, W, 3] RGB image (uint8)
            camera_intrinsics: [3, 3] camera K matrix
            depth: Optional [H, W] depth map

        Returns:
            List of detections, each containing:
                - bbox: [4] bounding box (x1, y1, x2, y2)
                - score: float confidence score
                - class_id: int class label
                - mask: [H, W] binary mask
                - nocs: [H, W, 3] NOCS coordinate map
                - pose: dict with R, t, s (if depth available)
        """
        # Preprocess image
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Forward pass
        outputs = self.model(image_tensor)

        # Post-process outputs
        detections = self._post_process(
            outputs,
            image.shape[:2],
            camera_intrinsics,
            depth,
        )

        return detections

    def _post_process(
        self,
        outputs: Dict,
        image_shape: tuple,
        camera_intrinsics: np.ndarray,
        depth: np.ndarray = None,
    ) -> List[Dict]:
        """Post-process model outputs to get final detections."""
        detections = []

        if len(outputs["proposals"]) == 0:
            return detections

        # Get predictions
        class_logits = outputs["class_logits"]  # [N, num_classes]
        mask_logits = outputs["mask_logits"]  # [N, num_classes, 28, 28]
        nocs_logits = outputs["nocs_logits"]  # [N, num_classes, 3, 32, 28, 28]
        proposals = torch.cat(outputs["proposals"], dim=0)  # [N, 4]

        # Get class predictions and scores
        class_probs = torch.softmax(class_logits, dim=1)
        scores, class_ids = class_probs.max(dim=1)

        # Filter by score and class (ignore background class 0)
        keep = (scores > self.score_threshold) & (class_ids > 0)

        if keep.sum() == 0:
            return detections

        # Filter
        proposals = proposals[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        mask_logits = mask_logits[keep]
        nocs_logits = nocs_logits[keep]

        # Process each detection
        H, W = image_shape

        for i in range(len(proposals)):
            bbox = proposals[i].cpu().numpy()
            score = scores[i].cpu().item()
            class_id = class_ids[i].cpu().item()

            # Get mask for this class
            mask_logit = mask_logits[i, class_id]  # [28, 28]
            mask_prob = torch.sigmoid(mask_logit).cpu().numpy()

            # Resize mask to bbox size
            x1, y1, x2, y2 = bbox.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)

            bbox_h = y2 - y1
            bbox_w = x2 - x1

            if bbox_h <= 0 or bbox_w <= 0:
                continue

            mask_resized = cv2.resize(mask_prob, (bbox_w, bbox_h))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            # Create full-size mask
            full_mask = np.zeros((H, W), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = mask_binary

            # Get NOCS coordinates for this class
            nocs_logit = nocs_logits[i, class_id]  # [3, 32, 28, 28]

            # Decode NOCS (convert bins to continuous values)
            nocs_probs = torch.softmax(nocs_logit, dim=1)  # [3, 32, 28, 28]
            bin_centers = (
                torch.arange(32, device=nocs_probs.device).float() + 0.5
            ) / 32
            bin_centers = bin_centers.view(1, 32, 1, 1)
            nocs_pred = (nocs_probs * bin_centers).sum(dim=1)  # [3, 28, 28]
            nocs_pred = nocs_pred.cpu().numpy().transpose(1, 2, 0)  # [28, 28, 3]

            # Resize NOCS to bbox size
            nocs_resized = cv2.resize(nocs_pred, (bbox_w, bbox_h))

            # Create full-size NOCS map
            full_nocs = np.zeros((H, W, 3), dtype=np.float32)
            full_nocs[y1:y2, x1:x2] = nocs_resized

            # Estimate pose if depth available
            pose = None
            if depth is not None:
                result = estimate_pose_from_nocs(
                    full_nocs,
                    depth,
                    full_mask,
                    camera_intrinsics,
                    use_ransac=True,
                )

                if result is not None:
                    R, t, s = result
                    pose = {"R": R, "t": t, "s": s}

            detections.append(
                {
                    "bbox": bbox,
                    "score": score,
                    "class_id": class_id,
                    "mask": full_mask,
                    "nocs": full_nocs,
                    "pose": pose,
                }
            )

        return detections

    def visualize_results(
        self,
        image: np.ndarray,
        detections: List[Dict],
        save_path: str = None,
    ):
        """Visualize detection results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # RGB with bboxes
        ax = axes[0]
        ax.imshow(image)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="green",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                y1 - 5,
                f"Class {det['class_id']}: {det['score']:.2f}",
                color="white",
                bbox=dict(facecolor="green", alpha=0.7),
            )
        ax.set_title("Detections")
        ax.axis("off")

        # Masks
        ax = axes[1]
        combined_mask = np.zeros_like(image)
        for i, det in enumerate(detections):
            color = plt.cm.tab10(i % 10)[:3]
            combined_mask[det["mask"] > 0] = (np.array(color) * 255).astype(np.uint8)
        ax.imshow(combined_mask)
        ax.set_title("Masks")
        ax.axis("off")

        # NOCS
        ax = axes[2]
        if len(detections) > 0:
            # Show NOCS for first detection
            ax.imshow(detections[0]["nocs"])
            ax.set_title("NOCS Coordinates")
        ax.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

        plt.close()


def main():
    parser = argparse.ArgumentParser(description="NOCS R-CNN Inference")

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--output", type=str, default="inference_output", help="Output directory"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.7, help="Detection score threshold"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image = np.array(Image.open(args.image).convert("RGB"))

    # Dummy camera intrinsics (replace with actual values)
    camera_intrinsics = np.array(
        [[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]]
    )

    # Create inference engine
    inference = NOCSInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        score_threshold=args.score_threshold,
    )

    # Run inference
    print(f"Running inference on {args.image}...")
    detections = inference.predict(image, camera_intrinsics)

    print(f"\nFound {len(detections)} detections:")
    for i, det in enumerate(detections):
        print(f"  Detection {i+1}:")
        print(f"    Class: {det['class_id']}")
        print(f"    Score: {det['score']:.3f}")
        print(f"    BBox: {det['bbox']}")
        if det["pose"] is not None:
            print(
                f"    Pose: R={det['pose']['R'].shape}, t={det['pose']['t']}, s={det['pose']['s']:.3f}"
            )

    # Visualize
    save_path = output_dir / f"{Path(args.image).stem}_result.png"
    inference.visualize_results(image, detections, save_path=str(save_path))

    # Save results
    results_path = output_dir / f"{Path(args.image).stem}_results.json"
    results = []
    for det in detections:
        result = {
            "bbox": det["bbox"].tolist(),
            "score": float(det["score"]),
            "class_id": int(det["class_id"]),
        }
        if det["pose"] is not None:
            result["pose"] = {
                "R": det["pose"]["R"].tolist(),
                "t": det["pose"]["t"].tolist(),
                "s": float(det["pose"]["s"]),
            }
        results.append(result)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
