"""
Estimate 6DoF pose from NOCS predictions using Umeyama algorithm.

Implements the pose estimation pipeline from the NOCS paper:
1. Extract NOCS coordinates for detected object
2. Back-project 2D pixels to 3D using camera intrinsics
3. Align predicted NOCS to canonical model using Umeyama
4. Apply RANSAC for outlier rejection
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def umeyama_alignment(
    src: np.ndarray, dst: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate similarity transformation (rotation, translation, scale)
    from src to dst using Umeyama's algorithm.

    Args:
        src: Source points [N, 3]
        dst: Destination points [N, 3]

    Returns:
        R: Rotation matrix [3, 3]
        t: Translation vector [3]
        s: Scale factor
    """
    assert src.shape == dst.shape
    assert src.shape[1] == 3

    num_points = src.shape[0]

    # Compute centroids
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)

    # Center the points
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Compute covariance matrix
    H = src_centered.T @ dst_centered / num_points

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute scale
    src_var = np.var(src_centered, axis=0).sum()
    s = np.trace(np.diag(S)) / src_var if src_var > 0 else 1.0

    # Compute translation
    t = dst_mean - s * R @ src_mean

    return R, t, s


def backproject_depth(
    depth_map: np.ndarray, mask: np.ndarray, K: np.ndarray
) -> np.ndarray:
    """
    Backproject depth map to 3D points.

    Args:
        depth_map: Depth values [H, W]
        mask: Valid pixel mask [H, W]
        K: Camera intrinsics [3, 3]

    Returns:
        points_3d: 3D points [N, 3] in camera coordinates
    """
    h, w = depth_map.shape

    # Get valid pixel coordinates
    v, u = np.where(mask > 0)

    if len(v) == 0:
        return np.zeros((0, 3))

    # Get depth values
    z = depth_map[v, u]

    # Backproject to 3D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_3d = np.stack([x, y, z], axis=1)

    return points_3d


def estimate_pose_from_nocs(
    nocs_map: np.ndarray,
    mask: np.ndarray,
    depth_map: np.ndarray,
    K: np.ndarray,
    scale: float = 1.0,
    use_ransac: bool = True,
    ransac_iterations: int = 1000,
    ransac_threshold: float = 0.01,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    Estimate 6DoF pose from NOCS prediction.

    Args:
        nocs_map: Predicted NOCS coordinates [H, W, 3], values in [0, 1]
        mask: Object mask [H, W], binary
        depth_map: Depth map [H, W], in meters
        K: Camera intrinsics [3, 3]
        scale: Object scale (from model metadata or prediction)
        use_ransac: Whether to use RANSAC for outlier rejection
        ransac_iterations: Number of RANSAC iterations
        ransac_threshold: RANSAC inlier threshold in meters

    Returns:
        R: Rotation matrix [3, 3] (or None if failed)
        t: Translation vector [3] (or None if failed)
        estimated_scale: Estimated scale (or None if failed)
    """
    # Resize NOCS map to match depth map if needed
    if nocs_map.shape[:2] != depth_map.shape:
        nocs_map = cv2.resize(nocs_map, (depth_map.shape[1], depth_map.shape[0]))

    if mask.shape != depth_map.shape:
        mask = cv2.resize(
            mask.astype(np.uint8), (depth_map.shape[1], depth_map.shape[0])
        )

    # Get valid pixels (inside mask and valid depth)
    valid_mask = (mask > 0) & (depth_map > 0)

    v, u = np.where(valid_mask)

    if len(v) < 10:  # Need minimum points
        logger.warning(f"Too few valid points: {len(v)}")
        return None, None, None

    # Get NOCS coordinates (source points in canonical space [0, 1]^3)
    nocs_coords = nocs_map[v, u]  # [N, 3]

    # Center NOCS to [-0.5, 0.5]^3
    nocs_coords_centered = nocs_coords - 0.5

    # Backproject to 3D camera coordinates (destination points)
    points_3d_camera = backproject_depth(depth_map, valid_mask, K)

    if len(points_3d_camera) < 10:
        logger.warning("Backprojection failed")
        return None, None, None

    # RANSAC loop (if enabled)
    if use_ransac and len(points_3d_camera) > 100:
        best_inliers = 0
        best_R = None
        best_t = None
        best_s = None

        for _ in range(ransac_iterations):
            # Sample random subset
            indices = np.random.choice(
                len(points_3d_camera),
                size=min(100, len(points_3d_camera)),
                replace=False,
            )

            src_sample = nocs_coords_centered[indices]
            dst_sample = points_3d_camera[indices]

            # Estimate transformation
            try:
                R, t, s = umeyama_alignment(src_sample, dst_sample)
            except:
                continue

            # Transform all source points
            src_transformed = s * (nocs_coords_centered @ R.T) + t

            # Count inliers
            distances = np.linalg.norm(src_transformed - points_3d_camera, axis=1)
            inliers = distances < ransac_threshold
            num_inliers = np.sum(inliers)

            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_R = R
                best_t = t
                best_s = s

        if best_R is None:
            logger.warning("RANSAC failed to find valid solution")
            return None, None, None

        # Refine with all inliers
        src_transformed = best_s * (nocs_coords_centered @ best_R.T) + best_t
        distances = np.linalg.norm(src_transformed - points_3d_camera, axis=1)
        inliers = distances < ransac_threshold

        if np.sum(inliers) < 10:
            return None, None, None

        R, t, s = umeyama_alignment(
            nocs_coords_centered[inliers], points_3d_camera[inliers]
        )
    else:
        # Direct estimation without RANSAC
        try:
            R, t, s = umeyama_alignment(nocs_coords_centered, points_3d_camera)
        except:
            return None, None, None

    # Validate rotation matrix
    if not is_valid_rotation_matrix(R):
        logger.warning("Invalid rotation matrix")
        return None, None, None

    return R, t, s


def is_valid_rotation_matrix(R: np.ndarray, tol: float = 1e-3) -> bool:
    """Check if matrix is a valid rotation matrix."""
    # Check orthogonality
    should_be_identity = R @ R.T
    identity = np.eye(3)
    if not np.allclose(should_be_identity, identity, atol=tol):
        return False

    # Check determinant
    if not np.isclose(np.linalg.det(R), 1.0, atol=tol):
        return False

    return True


def compute_rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    Compute rotation error in degrees.

    Args:
        R_pred: Predicted rotation [3, 3]
        R_gt: Ground truth rotation [3, 3]

    Returns:
        error: Rotation error in degrees
    """
    R_diff = R_pred @ R_gt.T
    trace = np.trace(R_diff)
    # Clamp to valid range for arccos
    trace = np.clip(trace, -1.0, 3.0)
    angle = np.arccos((trace - 1.0) / 2.0)
    return np.degrees(angle)


def compute_translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """
    Compute translation error in meters.

    Args:
        t_pred: Predicted translation [3]
        t_gt: Ground truth translation [3]

    Returns:
        error: Translation error in meters
    """
    return np.linalg.norm(t_pred - t_gt)


def compute_3d_iou(
    nocs_pred: np.ndarray,
    nocs_gt: np.ndarray,
    mask_pred: np.ndarray,
    mask_gt: np.ndarray,
) -> float:
    """
    Compute 3D IoU between predicted and ground truth NOCS maps.

    Args:
        nocs_pred: Predicted NOCS [H, W, 3]
        nocs_gt: Ground truth NOCS [H, W, 3]
        mask_pred: Predicted mask [H, W]
        mask_gt: Ground truth mask [H, W]

    Returns:
        iou: 3D IoU score
    """
    # Resize if needed
    if nocs_pred.shape[:2] != nocs_gt.shape[:2]:
        nocs_pred = cv2.resize(nocs_pred, (nocs_gt.shape[1], nocs_gt.shape[0]))
        mask_pred = cv2.resize(
            mask_pred.astype(np.uint8), (nocs_gt.shape[1], nocs_gt.shape[0])
        )

    # Use predicted and GT masks
    intersection_mask = (mask_pred > 0) & (mask_gt > 0)
    union_mask = (mask_pred > 0) | (mask_gt > 0)

    if np.sum(union_mask) == 0:
        return 0.0

    # Compute distance in NOCS space
    distances = np.linalg.norm(nocs_pred - nocs_gt, axis=2)

    # Threshold: consider as match if distance < some threshold (e.g., 0.1 in NOCS space)
    threshold = 0.1
    matches = (distances < threshold) & intersection_mask

    iou = np.sum(matches) / np.sum(union_mask)

    return iou
