"""
Inference script for NOCS R-CNN.

Supports:
- Single image inference
- HDF5 file inference (with GT comparison)
- Pose estimation with optional GT scale
- 3D coordinate frame visualization
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from sim2real_6dof.training.model.nocs_maskrcnn import NOCSMaskRCNN
from sim2real_6dof.training.utils.pose_estimation import (
    compute_rotation_error,
    compute_translation_error,
    umeyama_alignment,
)

# Default camera intrinsics (Kinect-like, matching data generation)
DEFAULT_FX = 572.4114
DEFAULT_FY = 573.57043
DEFAULT_CX = 325.2611
DEFAULT_CY = 242.04899


def project_axes(rvec, tvec, K, axis_length=0.1):
    """
    Project 3D coordinate axes onto the image plane.

    Args:
        rvec: [3, 1] or [3] rotation vector (Rodrigues format)
        tvec: [3, 1] or [3] translation vector
        K: [3, 3] camera intrinsic matrix
        axis_length: Length of axes in meters

    Returns:
        axes_2d: [4, 2] projected points [origin, X-end, Y-end, Z-end]
    """
    # Define 3D points: origin and 3 axis endpoints
    axes_3d = np.float32(
        [
            [0, 0, 0],  # Origin
            [axis_length, 0, 0],  # X-axis (red)
            [0, axis_length, 0],  # Y-axis (green)
            [0, 0, axis_length],  # Z-axis (blue)
        ]
    )

    # Project to 2D
    axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, None)
    axes_2d = axes_2d.reshape(-1, 2)

    return axes_2d


def draw_axes(image, axes_2d, thickness=2):
    """
    Draw coordinate frame axes on image.

    Args:
        image: RGB image
        axes_2d: [4, 2] projected points from project_axes
        thickness: Line thickness

    Returns:
        image with axes drawn
    """
    image = image.copy()
    origin = tuple(axes_2d[0].astype(int))

    # X-axis (red)
    x_end = tuple(axes_2d[1].astype(int))
    cv2.line(image, origin, x_end, (255, 0, 0), thickness)

    # Y-axis (green)
    y_end = tuple(axes_2d[2].astype(int))
    cv2.line(image, origin, y_end, (0, 255, 0), thickness)

    # Z-axis (blue)
    z_end = tuple(axes_2d[3].astype(int))
    cv2.line(image, origin, z_end, (0, 0, 255), thickness)

    return image


def decode_nocs_from_logits(nocs_logits, class_id, num_bins=32):
    """
    Decode NOCS coordinates from bin classification logits.

    Args:
        nocs_logits: [num_classes, 3, num_bins, H, W] logits
        class_id: int, which class to decode
        num_bins: number of bins

    Returns:
        nocs_map: [H, W, 3] NOCS coordinates in [0, 1]
    """
    # Select class
    class_logits = nocs_logits[class_id]  # [3, num_bins, H, W]

    # Softmax over bins
    probs = torch.softmax(class_logits, dim=1)  # [3, num_bins, H, W]

    # Bin centers
    bin_centers = (torch.arange(num_bins, device=probs.device).float() + 0.5) / num_bins
    bin_centers = bin_centers.view(1, num_bins, 1, 1)

    # Expected value (weighted sum)
    coords = (probs * bin_centers).sum(dim=1)  # [3, H, W]

    # Transpose to [H, W, 3]
    coords = coords.permute(1, 2, 0).cpu().numpy()

    return coords


def estimate_pose_from_nocs(
    nocs_map: np.ndarray,
    mask: np.ndarray,
    camera_intrinsics: np.ndarray,
    gt_scale: Optional[float] = None,
    min_points: int = 100,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Estimate 6DoF pose from NOCS map.

    Args:
        nocs_map: [H, W, 3] NOCS coordinates in [0, 1]
        mask: [H, W] binary mask
        camera_intrinsics: [3, 3] camera K matrix
        gt_scale: Optional ground truth scale in meters (alternative to depth)
        min_points: Minimum points required

    Returns:
        R: [3, 3] rotation matrix
        t: [3] translation vector
        s: scale factor
        (or None if failed)
    """
    # Get valid points
    valid_mask = mask > 0
    if valid_mask.sum() < min_points:
        return None

    # NOCS coordinates (canonical space centered at 0.5)
    nocs_coords = nocs_map[valid_mask] - 0.5  # Center at origin

    # Get 2D image coordinates
    ys, xs = np.where(valid_mask)

    # Estimate depth from bounding box size (if no GT scale provided)
    if gt_scale is None:
        bbox_height = ys.max() - ys.min()
        bbox_width = xs.max() - xs.min()
        bbox_size = max(bbox_height, bbox_width)

        # Heuristic: assume object is roughly 0.1m and estimate depth
        focal_length = (camera_intrinsics[0, 0] + camera_intrinsics[1, 1]) / 2
        estimated_object_size = 0.1  # meters
        estimated_depth = (focal_length * estimated_object_size) / bbox_size
    else:
        # Use GT scale to estimate depth
        bbox_height = ys.max() - ys.min()
        bbox_width = xs.max() - xs.min()
        bbox_size = max(bbox_height, bbox_width)

        focal_length = (camera_intrinsics[0, 0] + camera_intrinsics[1, 1]) / 2
        estimated_depth = (focal_length * gt_scale) / bbox_size

    # Backproject to 3D
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    x_3d = (xs - cx) * estimated_depth / fx
    y_3d = (ys - cy) * estimated_depth / fy
    z_3d = np.full_like(x_3d, estimated_depth)

    observed_points = np.stack([x_3d, y_3d, z_3d], axis=1)  # [N, 3]

    # Umeyama alignment
    try:
        R, t, s = umeyama_alignment(nocs_coords, observed_points, estimate_scale=True)

        # If GT scale provided, override estimated scale
        if gt_scale is not None:
            s = gt_scale
            # Recompute translation with fixed scale
            t = observed_points.mean(axis=0) - s * R @ nocs_coords.mean(axis=0)

        return R, t, s
    except:
        return None


class NOCSInference:
    """Inference wrapper for NOCS-enhanced Mask R-CNN."""

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
        self.num_bins = num_bins

        # Load model
        self.model = NOCSMaskRCNN(
            num_classes=num_classes,
            num_bins=num_bins,
            pretrained_backbone=False,
        ).to(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Iteration: {checkpoint.get('iteration', 'unknown')}")
        print(f"  Stage: {checkpoint.get('stage', 'unknown')}")

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        camera_intrinsics: np.ndarray,
        gt_scale: Optional[float] = None,
    ) -> List[Dict]:
        """
        Run inference on a single image.

        Args:
            image: [H, W, 3] RGB image (uint8)
            camera_intrinsics: [3, 3] camera K matrix
            gt_scale: Optional ground truth scale in meters

        Returns:
            List of detections with bbox, score, mask, nocs, pose
        """
        # Preprocess
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        # Forward pass
        outputs = self.model([img_tensor])
        detections = outputs["detections"][0]

        # Filter by score
        keep = detections["scores"] > self.score_threshold

        results = []
        for i in torch.where(keep)[0]:
            bbox = detections["boxes"][i].cpu().numpy()
            score = detections["scores"][i].cpu().item()
            label = detections["labels"][i].cpu().item()

            # Mask
            mask_prob = detections["masks"][i, 0].cpu().numpy()  # [H, W]
            mask = (mask_prob > 0.5).astype(np.uint8)

            # NOCS
            nocs_logits = detections["nocs_logits"][
                i
            ]  # [num_classes, 3, num_bins, 28, 28]
            nocs_map_28 = decode_nocs_from_logits(nocs_logits, label, self.num_bins)

            # Resize NOCS to image size
            H, W = image.shape[:2]
            nocs_map = cv2.resize(nocs_map_28, (W, H), interpolation=cv2.INTER_LINEAR)

            # Apply mask
            nocs_map = nocs_map * mask[:, :, np.newaxis]

            # Estimate pose
            pose_result = estimate_pose_from_nocs(
                nocs_map,
                mask,
                camera_intrinsics,
                gt_scale=gt_scale,
            )

            if pose_result is not None:
                R, t, s = pose_result
                pose = {"R": R, "t": t, "s": s}
            else:
                pose = None

            results.append(
                {
                    "bbox": bbox,
                    "score": score,
                    "label": label,
                    "mask": mask,
                    "nocs": nocs_map,
                    "pose": pose,
                }
            )

        return results

    def visualize(
        self,
        image: np.ndarray,
        detections: List[Dict],
        camera_intrinsics: np.ndarray,
        gt_pose: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Visualize detections with bboxes, masks, NOCS, and pose axes.

        Returns:
            [H*2, W*2, 3] visualization grid
        """
        H, W = image.shape[:2]

        # Create 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # 1. RGB with bboxes and pose axes
        ax = axes[0, 0]
        vis_img = image.copy()

        for det in detections:
            # Draw bbox
            x1, y1, x2, y2 = det["bbox"].astype(int)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label_text = f"Class {det['label']}: {det['score']:.2f}"
            cv2.putText(
                vis_img,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Draw pose axes
            if det["pose"] is not None:
                R = det["pose"]["R"]
                t = det["pose"]["t"]

                # Convert rotation matrix to Rodrigues
                rvec, _ = cv2.Rodrigues(R)
                tvec = t.reshape(3, 1)

                axes_2d = project_axes(rvec, tvec, camera_intrinsics, axis_length=0.05)
                vis_img = draw_axes(vis_img, axes_2d, thickness=2)

        ax.imshow(vis_img)
        ax.set_title("Detections with Pose")
        ax.axis("off")

        # 2. Masks
        ax = axes[0, 1]
        mask_vis = np.zeros_like(image)
        for i, det in enumerate(detections):
            color = plt.cm.tab10(i % 10)[:3]
            mask_vis[det["mask"] > 0] = (np.array(color) * 255).astype(np.uint8)
        ax.imshow(mask_vis)
        ax.set_title("Instance Masks")
        ax.axis("off")

        # 3. NOCS maps
        ax = axes[1, 0]
        if len(detections) > 0:
            ax.imshow(detections[0]["nocs"])
            ax.set_title("NOCS Coordinates")
        ax.axis("off")

        # 4. Pose comparison (if GT available)
        ax = axes[1, 1]
        if gt_pose is not None and len(detections) > 0:
            det = detections[0]
            if det["pose"] is not None:
                R_pred = det["pose"]["R"]
                t_pred = det["pose"]["t"]
                s_pred = det["pose"]["s"]

                R_gt = gt_pose["R"]
                t_gt = gt_pose["t"]
                s_gt = gt_pose["s"]

                rot_err = compute_rotation_error(R_pred, R_gt)
                trans_err = compute_translation_error(t_pred, t_gt)
                scale_err = abs(s_pred - s_gt) / s_gt * 100

                info_text = f"Pose Errors:\n"
                info_text += f"Rotation: {rot_err:.2f}°\n"
                info_text += f"Translation: {trans_err:.4f}m\n"
                info_text += f"Scale: {scale_err:.2f}%\n\n"
                info_text += f"Predicted:\n"
                info_text += (
                    f"  t: [{t_pred[0]:.3f}, {t_pred[1]:.3f}, {t_pred[2]:.3f}]\n"
                )
                info_text += f"  s: {s_pred:.3f}\n\n"
                info_text += f"Ground Truth:\n"
                info_text += f"  t: [{t_gt[0]:.3f}, {t_gt[1]:.3f}, {t_gt[2]:.3f}]\n"
                info_text += f"  s: {s_gt:.3f}"

                ax.text(
                    0.05,
                    0.95,
                    info_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                    family="monospace",
                )
        ax.set_title("Pose Comparison" if gt_pose else "Info")
        ax.axis("off")

        plt.tight_layout()

        # Convert to numpy array
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        vis_array = vis_array.reshape((-1, 4))[:, 1:]
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return vis_array


def main():
    parser = argparse.ArgumentParser(description="NOCS R-CNN Inference")

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input image or HDF5 file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inference_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.7, help="Detection score threshold"
    )
    parser.add_argument("--show", action="store_true", help="Show results in window")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save results to file"
    )

    # Camera intrinsics (optional overrides)
    parser.add_argument(
        "--fx", type=float, default=None, help="Focal length X (overrides default)"
    )
    parser.add_argument(
        "--fy", type=float, default=None, help="Focal length Y (overrides default)"
    )
    parser.add_argument(
        "--cx", type=float, default=None, help="Principal point X (overrides default)"
    )
    parser.add_argument(
        "--cy", type=float, default=None, help="Principal point Y (overrides default)"
    )
    parser.add_argument(
        "--override-intrinsics",
        action="store_true",
        help="Override HDF5 intrinsics with command-line values (only applies to HDF5 input)",
    )

    # Ground truth scale (alternative to depth)
    parser.add_argument(
        "--gt-scale",
        type=float,
        default=None,
        help="Ground truth object scale in meters",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine input type
    input_path = Path(args.input)
    is_hdf5 = input_path.suffix.lower() in [".hdf5", ".h5"]

    # Load data
    if is_hdf5:
        print(f"Loading HDF5 file: {input_path}")
        with h5py.File(input_path, "r") as f:
            # Image
            image = np.array(f["colors"])
            if image.shape[-1] == 4:
                image = image[:, :, :3]

            # Camera intrinsics
            if args.override_intrinsics or any([args.fx, args.fy, args.cx, args.cy]):
                # Use command-line specified intrinsics (or defaults)
                K = np.array(
                    [
                        [args.fx or DEFAULT_FX, 0, args.cx or DEFAULT_CX],
                        [0, args.fy or DEFAULT_FY, args.cy or DEFAULT_CY],
                        [0, 0, 1],
                    ],
                    dtype=np.float32,
                )
                print("Using command-line specified camera intrinsics")
            else:
                # Use intrinsics from HDF5 file
                K = np.array(f["camera_intrinsics"], dtype=np.float32)
                print("Using camera intrinsics from HDF5 file")

            # Ground truth pose (for comparison)
            R_gt = np.array(f["object_to_camera_rotation"])
            t_gt = np.array(f["object_to_camera_translation"])

            # Get scale from metadata
            metadata_raw = f["metadata"][()]
            if isinstance(metadata_raw, bytes):
                metadata_str = metadata_raw.decode("utf-8")
            elif isinstance(metadata_raw, np.bytes_):
                metadata_str = metadata_raw.decode("utf-8")
            else:
                metadata_str = str(metadata_raw)
            metadata = json.loads(metadata_str)
            s_gt = metadata.get("object_scale", 1.0)

            gt_pose = {"R": R_gt, "t": t_gt, "s": s_gt}

            # Use GT scale if not overridden
            if args.gt_scale is None:
                gt_scale = s_gt
            else:
                gt_scale = args.gt_scale
    else:
        print(f"Loading image: {input_path}")
        image = np.array(Image.open(input_path).convert("RGB"))

        # Default camera intrinsics
        K = np.array(
            [
                [args.fx or DEFAULT_FX, 0, args.cx or DEFAULT_CX],
                [0, args.fy or DEFAULT_FY, args.cy or DEFAULT_CY],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        gt_pose = None
        gt_scale = args.gt_scale

    # Create inference engine
    print(f"Loading model from {args.checkpoint}...")
    inference = NOCSInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        score_threshold=args.score_threshold,
    )

    # Run inference
    print("Running inference...")
    detections = inference.predict(image, K, gt_scale=gt_scale)

    print(f"\nFound {len(detections)} detections")
    for i, det in enumerate(detections):
        print(f"  Detection {i+1}:")
        print(f"    Label: {det['label']}")
        print(f"    Score: {det['score']:.3f}")
        print(f"    BBox: {det['bbox']}")
        if det["pose"] is not None:
            print(f"    Pose:")
            print(f"      Translation: {det['pose']['t']}")
            print(f"      Scale: {det['pose']['s']:.4f}")

    # Compare with GT if available
    if gt_pose is not None and len(detections) > 0:
        det = detections[0]
        if det["pose"] is not None:
            R_pred = det["pose"]["R"]
            t_pred = det["pose"]["t"]
            s_pred = det["pose"]["s"]

            rot_err = compute_rotation_error(R_pred, gt_pose["R"])
            trans_err = compute_translation_error(t_pred, gt_pose["t"])
            scale_err = abs(s_pred - gt_pose["s"]) / gt_pose["s"] * 100

            print(f"\n  Pose Errors vs Ground Truth:")
            print(f"    Rotation: {rot_err:.2f}°")
            print(f"    Translation: {trans_err:.4f}m")
            print(f"    Scale: {scale_err:.2f}%")

    # Visualize
    print("\nGenerating visualization...")
    vis_image = inference.visualize(image, detections, K, gt_pose=gt_pose)

    # Save
    if args.save:
        vis_path = output_dir / f"{input_path.stem}_result.png"
        Image.fromarray(vis_image).save(vis_path)
        print(f"Saved visualization to {vis_path}")

        # Save pose JSON
        if len(detections) > 0 and detections[0]["pose"] is not None:
            pose_path = output_dir / f"{input_path.stem}_pose.json"
            pose_data = {
                "R": detections[0]["pose"]["R"].tolist(),
                "t": detections[0]["pose"]["t"].tolist(),
                "s": float(detections[0]["pose"]["s"]),
            }
            if gt_pose is not None:
                pose_data["ground_truth"] = {
                    "R": gt_pose["R"].tolist(),
                    "t": gt_pose["t"].tolist(),
                    "s": float(gt_pose["s"]),
                }
                pose_data["errors"] = {
                    "rotation_deg": float(rot_err),
                    "translation_m": float(trans_err),
                    "scale_percent": float(scale_err),
                }

            with open(pose_path, "w") as f:
                json.dump(pose_data, f, indent=2)
            print(f"Saved pose data to {pose_path}")

    # Show
    if args.show:
        plt.figure(figsize=(12, 12))
        plt.imshow(vis_image)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
