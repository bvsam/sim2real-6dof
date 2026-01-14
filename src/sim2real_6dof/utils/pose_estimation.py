"""
Pose estimation utilities for NOCS.

Implements Umeyama's algorithm to recover 6DoF pose and scale
from corresponding point sets (NOCS predictions vs canonical model).
"""

from typing import Optional, Tuple

import numpy as np
import torch


def umeyama_alignment(
    source_points: np.ndarray, target_points: np.ndarray, estimate_scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Umeyama algorithm for rigid alignment between two point sets.

    Computes optimal rotation R, translation t, and scale s such that:
        target = s * R @ source + t

    Args:
        source_points: [N, 3] source points (e.g., NOCS coordinates)
        target_points: [N, 3] target points (e.g., observed 3D points)
        estimate_scale: Whether to estimate scale (True for NOCS)

    Returns:
        R: [3, 3] rotation matrix
        t: [3] translation vector
        s: scalar scale factor

    Reference: "Least-squares estimation of transformation parameters
               between two point patterns" - Umeyama 1991
    """
    assert source_points.shape == target_points.shape
    assert source_points.shape[1] == 3

    num_points = source_points.shape[0]

    # Compute centroids
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    # Center the points
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # Compute the covariance matrix
    H = source_centered.T @ target_centered / num_points

    # SVD
    U, D, Vt = np.linalg.svd(H)

    # Compute rotation
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1, 1, d])
    R = Vt.T @ S @ U.T

    # Ensure proper rotation matrix (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute scale
    if estimate_scale:
        source_variance = np.var(source_centered, axis=0).sum()
        s = np.trace(np.diag(D) @ S) / source_variance
    else:
        s = 1.0

    # Compute translation
    t = target_centroid - s * R @ source_centroid

    return R, t, s


def estimate_pose_from_nocs(
    nocs_map: np.ndarray,
    depth_map: np.ndarray,
    mask: np.ndarray,
    camera_intrinsics: np.ndarray,
    min_points: int = 100,
    use_ransac: bool = True,
    ransac_iterations: int = 1000,
    ransac_threshold: float = 0.01,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Estimate 6DoF pose from NOCS prediction.

    Args:
        nocs_map: [H, W, 3] predicted NOCS coordinates in [0, 1]
        depth_map: [H, W] depth map (if available, otherwise use dummy depth)
        mask: [H, W] binary mask of object pixels
        camera_intrinsics: [3, 3] camera intrinsic matrix
        min_points: Minimum number of points required
        use_ransac: Whether to use RANSAC for outlier rejection
        ransac_iterations: Number of RANSAC iterations
        ransac_threshold: RANSAC inlier threshold

    Returns:
        R: [3, 3] rotation matrix (None if failed)
        t: [3] translation vector
        s: scale factor
    """
    # Get valid points
    valid_mask = mask > 0
    if valid_mask.sum() < min_points:
        return None

    # NOCS coordinates (source points in canonical space [0, 1]^3)
    nocs_coords = nocs_map[valid_mask]  # [N, 3]

    # Observed 3D points in camera space (target points)
    # If depth is not available, we can't directly get 3D points
    # For NOCS, we use the mask bbox center as a proxy
    ys, xs = np.where(valid_mask)

    if depth_map is not None and depth_map[valid_mask].max() > 0:
        # Use actual depth
        depths = depth_map[valid_mask]

        # Backproject to 3D using camera intrinsics
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        x_3d = (xs - cx) * depths / fx
        y_3d = (ys - cy) * depths / fy
        z_3d = depths

        observed_points = np.stack([x_3d, y_3d, z_3d], axis=1)  # [N, 3]
    else:
        # No depth available - use a simplified approach
        # This is common during inference on RGB-only data
        # We'll estimate depth from the NOCS coordinates and mask size

        # Estimate object center in image
        center_x = xs.mean()
        center_y = ys.mean()

        # Estimate depth from bounding box size (heuristic)
        bbox_height = ys.max() - ys.min()
        bbox_width = xs.max() - xs.min()
        bbox_size = max(bbox_height, bbox_width)

        # Assume canonical object has unit size, estimate depth
        focal_length = (camera_intrinsics[0, 0] + camera_intrinsics[1, 1]) / 2
        estimated_depth = focal_length / bbox_size  # Rough estimate

        # Create pseudo-3D points
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        x_3d = (xs - cx) * estimated_depth / fx
        y_3d = (ys - cy) * estimated_depth / fy
        z_3d = np.full_like(x_3d, estimated_depth)

        observed_points = np.stack([x_3d, y_3d, z_3d], axis=1)

    # Center NOCS coordinates (canonical space is [0,1]^3, center at 0.5)
    nocs_centered = nocs_coords - 0.5

    if use_ransac and len(nocs_coords) > min_points * 2:
        # RANSAC for robust estimation
        best_inliers = 0
        best_R, best_t, best_s = None, None, None

        for _ in range(ransac_iterations):
            # Sample random subset
            sample_size = min(min_points, len(nocs_coords) // 2)
            indices = np.random.choice(len(nocs_coords), sample_size, replace=False)

            # Estimate pose from subset
            try:
                R, t, s = umeyama_alignment(
                    nocs_centered[indices],
                    observed_points[indices],
                    estimate_scale=True,
                )
            except:
                continue

            # Count inliers
            predicted = s * (R @ nocs_centered.T).T + t
            errors = np.linalg.norm(predicted - observed_points, axis=1)
            inliers = errors < ransac_threshold
            num_inliers = inliers.sum()

            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_R, best_t, best_s = R, t, s

        if best_R is None:
            return None

        # Refine with all inliers
        predicted = best_s * (best_R @ nocs_centered.T).T + best_t
        errors = np.linalg.norm(predicted - observed_points, axis=1)
        inliers = errors < ransac_threshold

        if inliers.sum() >= min_points:
            R, t, s = umeyama_alignment(
                nocs_centered[inliers], observed_points[inliers], estimate_scale=True
            )
        else:
            R, t, s = best_R, best_t, best_s
    else:
        # Direct estimation without RANSAC
        try:
            R, t, s = umeyama_alignment(
                nocs_centered, observed_points, estimate_scale=True
            )
        except:
            return None

    return R, t, s


def compute_rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    Compute rotation error in degrees.

    Args:
        R_pred: [3, 3] predicted rotation
        R_gt: [3, 3] ground truth rotation

    Returns:
        error: Rotation error in degrees
    """
    R_diff = R_pred @ R_gt.T
    trace = np.trace(R_diff)

    # Clamp for numerical stability
    trace = np.clip(trace, -1, 3)

    # Angle of rotation
    theta = np.arccos((trace - 1) / 2)

    return np.degrees(theta)


def compute_translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """
    Compute translation error (Euclidean distance).

    Args:
        t_pred: [3] predicted translation
        t_gt: [3] ground truth translation

    Returns:
        error: Translation error (L2 norm)
    """
    return np.linalg.norm(t_pred - t_gt)


def compute_scale_error(s_pred: float, s_gt: float) -> float:
    """
    Compute scale error (absolute percentage error).

    Args:
        s_pred: predicted scale
        s_gt: ground truth scale

    Returns:
        error: Absolute percentage error
    """
    return abs(s_pred - s_gt) / s_gt * 100
