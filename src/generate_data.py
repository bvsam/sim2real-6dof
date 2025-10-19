#!/usr/bin/env python3

import blenderproc as bproc  # isort: skip

"""
Synthetic 6DoF Mug Pose Estimation Data Generator

Generates training data for NOCS-based category-level 6DoF pose estimation.
Uses BlenderProc to render mugs with domain randomization, COCO backgrounds,
and generates ground truth: NOCS maps, masks, bboxes, keypoint heatmaps, and poses.
"""
import argparse
import json
import random
from pathlib import Path

import bpy
import numpy as np
import tqdm
from mathutils import Matrix
from PIL import Image

# =======================================================================================
# CONFIGURATION CONSTANTS
# =======================================================================================

# Default paths (can be overridden by CLI args)
DEFAULT_BASE_DIR = Path("/home/blender/workspace")
DEFAULT_MODEL_DIR = DEFAULT_BASE_DIR / "data" / "ModelNet40_canonicalized"
DEFAULT_KEYPOINT_PATH = DEFAULT_BASE_DIR / "annotations" / "canonical_keypoints.json"
DEFAULT_COCO_DIR = DEFAULT_BASE_DIR / "data" / "train2017"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "output"
DEFAULT_DEBUG_DIR = DEFAULT_BASE_DIR / "debug"
DEFAULT_TEXTURE_DIR = DEFAULT_BASE_DIR / "data" / "dtd" / "images"

DEFAULT_NUM_SAMPLES = 5

# Image resolution
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480

# Camera parameters
CAMERA_RADIUS_MIN = 0.25
CAMERA_RADIUS_MAX = 1.0
CAMERA_ELEVATION_MIN = -80  # default: -90 degrees
CAMERA_ELEVATION_MAX = 80  # default: 90 degrees
CAMERA_INPLANE_ROT_RANGE = 0.75  # radians

# Base camera intrinsics (Kinect-like camera)
BASE_FX = 572.4114
BASE_FY = 573.57043
BASE_CX = 325.2611
BASE_CY = 242.04899
INTRINSICS_VARIATION_RANGE = (0.97, 1.03)  # Focal length scale factor
PRINCIPAL_POINT_VARIATION = 10  # pixels

# Object parameters
OBJECT_BASE_SCALE = 0.1
OBJECT_SCALE_VARIATION = (0.85, 1.15)
OBJECT_ROTATION_RANGE_X = (-np.pi / 3, np.pi / 3)
OBJECT_ROTATION_RANGE_Y = (-np.pi / 3, np.pi / 3)
OBJECT_ROTATION_RANGE_Z = (-np.pi * 2, np.pi * 2)
OBJECT_LOCATION_RANGE_X = (-0.5, 0.5)
OBJECT_LOCATION_RANGE_Y = (-0.5, 0.5)
OBJECT_LOCATION_RANGE_Z = (-0.3, 0.3)

# Keypoint heatmap parameters
HEATMAP_SIZE = 64
HEATMAP_SIGMA = 2.0
KEYPOINT_NAMES = [
    "handle_top",
    "handle_bottom",
    "rim_center",
    "base_center",
    "rim_front",
    "rim_back",
    "rim_left",
    "rim_right",
]

# Domain randomization probabilities
PROB_DISTRACTOR_OBJECTS = 0.4
PROB_MOTION_BLUR = 0.05
PROB_DEPTH_OF_FIELD = 0.15
PROB_COLOR_SHIFT = 0.15
PROB_SUBSURFACE = 0.3

# Distractor parameters
NUM_DISTRACTORS_RANGE = (1, 3)
DISTRACTOR_SCALE_RANGE = (0.05, 0.3)
# Distractor placement (shell around mug)
DISTRACTOR_RADIUS_MIN = 0.25
DISTRACTOR_RADIUS_MAX = 0.50
DISTRACTOR_HEIGHT_OFFSET_RANGE = (-0.3, 0.3)

# Lighting parameters
NUM_LIGHTS_RANGE = (2, 4)
LIGHT_TYPES = ["SUN", "POINT", "SPOT"]
LIGHT_RADIUS_RANGE = (1.0, 4.0)
LIGHT_COLOR_RANGE = (0.8, 1.0)
LIGHT_ENERGY_SUN = (0.5, 2.0)
LIGHT_ENERGY_POINT = (40, 200)
LIGHT_ENERGY_SPOT = (40, 200)
AMBIENT_LIGHT_LOCATION = [0, 0, OBJECT_BASE_SCALE * 25]
AMBIENT_ENERGY_RANGE = (0.1, 0.3)

# Material randomization
MATERIAL_TYPE_PROBS = [0.15, 0.15, 0.7]  # PBR, solid, textured
MATERIAL_COLOR_RANGE = (0.1, 0.9)
MATERIAL_ROUGHNESS_RANGE = (0.0, 1.0)
MATERIAL_METALLIC_RANGE = (0.0, 1.0)

# Texture parameters
PROCEDURAL_TEXTURE_TYPES = ["stripes", "checker"]
TEXTURE_APPLICATION_PROB = 0.7  # Probability of using image texture vs procedural

# Sensor noise parameters
SHOT_NOISE_RANGE = (0.005, 0.02)
READ_NOISE_RANGE = (0.002, 0.01)
COLOR_SHIFT_RANGE = (0.97, 1.03)

# Motion blur parameters
MOTION_BLUR_SHUTTER_RANGE = (0.1, 0.5)

# Depth of field parameters
DOF_APERTURE_RANGE = (1.0, 4.0)

# Validation tolerances
ROTATION_ORTHOGONALITY_TOLERANCE = 1e-5
ROTATION_DETERMINANT_TOLERANCE = 1e-5

# Negative samples
NEGATIVE_SAMPLE_RATIO = 0.10


# =======================================================================================
# HELPER FUNCTIONS
# =======================================================================================


def save_debug_visualization(
    rgb_image,
    kps_2d,
    visibility,
    bbox,
    mask,
    nocs_map,
    heatmaps,
    sample_idx,
    model_name,
    save_keypoints=False,
):
    """Save debug visualization with keypoints, masks, and heatmaps."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # RGB with keypoints and bbox
    ax = axes[0, 0]
    ax.imshow(rgb_image)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(KEYPOINT_NAMES)))

    for i, (kp, vis, name, color) in enumerate(
        zip(kps_2d, visibility, KEYPOINT_NAMES, colors)
    ):
        if not save_keypoints:
            break
        if vis == 2:  # Visible
            ax.scatter(
                kp[0],
                kp[1],
                c=[color],
                s=50,
                marker="o",
                edgecolors="white",
                linewidths=2,
            )
            # ax.text(
            #     kp[0] + 5,
            #     kp[1],
            #     f"{i}:{name[:3]}",
            #     fontsize=7,
            #     color="white",
            #     bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
            # )
        elif vis == 1:  # Occluded
            ax.scatter(kp[0], kp[1], c=[color], s=30, marker="x", linewidths=2)

    if bbox is not None and bbox[0] >= 0:
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
        )
        ax.add_patch(rect)

    ax.set_title("RGB with Keypoints & BBox")
    ax.axis("off")

    # Instance mask
    ax = axes[0, 1]
    ax.imshow(mask, cmap="gray")
    ax.set_title("Instance Mask")
    ax.axis("off")

    # NOCS map
    ax = axes[0, 2]
    ax.imshow(nocs_map[:, :, :3])  # Only RGB channels
    ax.set_title("NOCS Map")
    ax.axis("off")

    # First 3 heatmaps
    for i in range(min(3, len(heatmaps))):
        ax = axes[1, i]
        ax.imshow(heatmaps[i], cmap="hot")
        ax.set_title(f"{KEYPOINT_NAMES[i]}")
        ax.axis("off")

    plt.suptitle(f"Sample {sample_idx}: {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    debug_path = Path(args.debug_dir) / f"sample_{sample_idx:05d}.png"
    print(f"Saving debug image to {debug_path}")
    plt.savefig(debug_path, dpi=120, bbox_inches="tight")
    plt.close()

    # Save the RGB image separately
    image = Image.fromarray(rgb_image)
    debug_rgb_image_path = Path(args.debug_dir) / f"sample_{sample_idx:05d}_rgb.png"
    print(f"Saving debug RGB image to {debug_rgb_image_path}")
    image.save(debug_rgb_image_path)


def generate_gaussian_heatmap(center, size, sigma):
    """Generate 2D Gaussian heatmap centered at given point."""
    x = np.arange(0, size, 1, float)
    y = np.arange(0, size, 1, float)[:, np.newaxis]

    cx = np.clip(center[0], 0, size - 1)
    cy = np.clip(center[1], 0, size - 1)

    heatmap = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap

    return heatmap.astype(np.float32)


def get_bounding_box_from_mask(mask):
    """Extract bounding box from binary mask."""
    points = np.column_stack(np.where(mask > 0))
    if len(points) == 0:
        return None

    y_min, x_min = points.min(axis=0)
    y_max, x_max = points.max(axis=0)

    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def compute_object_to_camera_transform(obj, camera_pose):
    """Compute 4x4 transformation from object to camera coordinates."""
    obj_to_world = np.array(obj.get_local2world_mat())
    world_to_camera = np.linalg.inv(camera_pose)
    obj_to_camera = world_to_camera @ obj_to_world

    # Orthogonalize rotation matrix
    R = obj_to_camera[:3, :3]
    t = obj_to_camera[:3, 3]

    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt

    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt

    obj_to_camera[:3, :3] = R_ortho
    obj_to_camera[:3, 3] = t

    return obj_to_camera


def add_sensor_noise(rgb_image):
    """Add realistic sensor noise to RGB image."""
    img_float = rgb_image.astype(np.float32) / 255.0
    num_channels = img_float.shape[2]

    # Separate RGB and alpha
    if num_channels == 4:
        rgb = img_float[:, :, :3]
        alpha = img_float[:, :, 3:4]
    else:
        rgb = img_float
        alpha = None

    # Shot noise
    shot_noise_strength = np.random.uniform(*SHOT_NOISE_RANGE)
    shot_noise = (
        np.random.normal(0, 1, rgb.shape) * np.sqrt(rgb + 0.01) * shot_noise_strength
    )
    rgb_noisy = rgb + shot_noise

    # Read noise
    read_noise_std = np.random.uniform(*READ_NOISE_RANGE)
    read_noise = np.random.normal(0, read_noise_std, rgb.shape)
    rgb_noisy = rgb_noisy + read_noise

    # Color shift
    if np.random.random() < PROB_COLOR_SHIFT:
        color_shift = np.random.uniform(*COLOR_SHIFT_RANGE, size=(1, 1, 3))
        rgb_noisy = rgb_noisy * color_shift

    rgb_noisy = np.clip(rgb_noisy, 0, 1)

    # Recombine with alpha
    if alpha is not None:
        img_noisy = np.concatenate([rgb_noisy, alpha], axis=2)
    else:
        img_noisy = rgb_noisy

    return (img_noisy * 255).astype(rgb_image.dtype)


def add_camera_effects(obj):
    """Add motion blur and depth of field effects."""
    # Motion blur
    if np.random.random() < PROB_MOTION_BLUR:
        bpy.context.scene.render.use_motion_blur = True
        bpy.context.scene.render.motion_blur_shutter = np.random.uniform(
            *MOTION_BLUR_SHUTTER_RANGE
        )
    else:
        bpy.context.scene.render.use_motion_blur = False

    # Depth of field
    cam = bpy.context.scene.camera
    if np.random.random() < PROB_DEPTH_OF_FIELD:
        poi = obj.get_location()
        cam_location = np.array(cam.location)
        distance_to_poi = np.linalg.norm(cam_location - poi)

        cam.data.dof.use_dof = True
        cam.data.dof.focus_distance = distance_to_poi
        cam.data.dof.aperture_fstop = np.random.uniform(*DOF_APERTURE_RANGE)
    else:
        cam.data.dof.use_dof = False


def generate_camera_intrinsics(image_width, image_height, args, add_variation=True):
    """Generate camera intrinsic matrix with optional variation, scaled to resolution."""
    # If intrinsics manually specified, use those
    fx = args.fx if args.fx is not None else BASE_FX
    fy = args.fy if args.fy is not None else BASE_FY
    cx = args.cx if args.cx is not None else BASE_CX
    cy = args.cy if args.cy is not None else BASE_CY

    # If all intrinsics aren't manually specified, scale them with resolution
    if not (
        args.fx is not None
        and args.fy is not None
        and args.cx is not None
        and args.cy is not None
    ):
        # Scale intrinsics based on resolution (BASE intrinsics are for 640x480)
        scale_x = image_width / DEFAULT_IMAGE_WIDTH
        scale_y = image_height / DEFAULT_IMAGE_HEIGHT
        fx = fx * scale_x
        fy = fy * scale_y
        cx = cx * scale_x
        cy = cy * scale_y

    if add_variation:
        focal_scale = np.random.uniform(*INTRINSICS_VARIATION_RANGE)
        principal_point_scale = np.random.uniform(
            -PRINCIPAL_POINT_VARIATION, PRINCIPAL_POINT_VARIATION
        )
        fx *= focal_scale
        fy *= focal_scale
        cx += principal_point_scale
        cy += principal_point_scale

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    return K


def get_keypoints_and_visibility(
    obj, ordered_kps_3d_local, camera_pose, K_matrix, image_width, image_height, frame=0
):
    """Project 3D keypoints to 2D and determine visibility."""
    bproc.camera.set_intrinsics_from_K_matrix(K_matrix, image_width, image_height)

    # Transform keypoints to world space
    obj_to_world = np.array(obj.get_local2world_mat())
    kps_3d_local_h = np.column_stack(
        [ordered_kps_3d_local, np.ones(len(ordered_kps_3d_local))]
    )
    kps_3d_world = ((obj_to_world @ kps_3d_local_h.T).T)[:, :3]

    # Project to 2D
    kps_2d_raw = bproc.camera.project_points(kps_3d_world, frame)

    # Check visibility
    visibility = []
    camera_location = camera_pose[:3, 3]

    for kp_world, kp_2d in zip(kps_3d_world, kps_2d_raw):
        # Out of frame check
        if not (0 <= kp_2d[0] < image_width and 0 <= kp_2d[1] < image_height):
            visibility.append(0)
            continue

        # Ray cast for occlusion
        direction = kp_world - camera_location
        distance_to_kp = np.linalg.norm(direction)
        direction_normalized = direction / distance_to_kp

        hit, hit_location, _, _, hit_object, _ = bproc.object.scene_ray_cast(
            camera_location, direction_normalized, max_distance=distance_to_kp + 1.0
        )

        if hit and hit_object is not None:
            hit_distance = np.linalg.norm(hit_location - camera_location)
            is_visible = (hit_distance >= distance_to_kp - 0.05) and (hit_object == obj)
        else:
            is_visible = True

        visibility.append(2 if is_visible else 1)

    return kps_2d_raw, np.array(visibility)


def add_distractor_objects(mug_location):
    """
    Add random primitive objects as distractors in a shell around the mug.

    Args:
        mug_location: 3D position of the mug (numpy array or list)

    Returns:
        List of distractor objects
    """
    num_distractors = np.random.randint(*NUM_DISTRACTORS_RANGE)
    distractors = []

    mug_pos = np.array(mug_location)

    for _ in range(num_distractors):
        primitive_type = np.random.choice(["cube", "sphere", "cylinder", "cone"])

        if primitive_type == "cube":
            distractor = bproc.object.create_primitive("CUBE")
            scale = np.random.uniform(*DISTRACTOR_SCALE_RANGE, 3)
        elif primitive_type == "sphere":
            distractor = bproc.object.create_primitive("SPHERE")
            scale = np.random.uniform(*DISTRACTOR_SCALE_RANGE, 3)
        elif primitive_type == "cylinder":
            distractor = bproc.object.create_primitive("CYLINDER")
            scale_xy = np.random.uniform(
                DISTRACTOR_SCALE_RANGE[0], DISTRACTOR_SCALE_RANGE[1]
            )
            scale_z = np.random.uniform(
                DISTRACTOR_SCALE_RANGE[0] * 2, DISTRACTOR_SCALE_RANGE[1] * 3
            )
            scale = [scale_xy, scale_xy, scale_z]
        else:  # cone
            distractor = bproc.object.create_primitive("CONE")
            scale = np.random.uniform(*DISTRACTOR_SCALE_RANGE, 3)

        distractor.set_scale(scale)

        # Place in a shell around the mug
        # Sample random point on horizontal circle around mug
        radius = np.random.uniform(DISTRACTOR_RADIUS_MIN, DISTRACTOR_RADIUS_MAX)
        angle = np.random.uniform(0, 2 * np.pi)

        # Horizontal offset from mug
        offset_x = radius * np.cos(angle)
        offset_y = radius * np.sin(angle)

        # Vertical offset (keep near table surface)
        offset_z = np.random.uniform(*DISTRACTOR_HEIGHT_OFFSET_RANGE)

        distractor_location = mug_pos + np.array([offset_x, offset_y, offset_z])
        distractor.set_location(distractor_location)

        distractor.set_rotation_euler(
            np.random.uniform([0, 0, 0], [np.pi * 2, np.pi * 2, np.pi * 2])
        )

        # Random material
        material = bproc.material.create(name=f"distractor_mat_{_}")
        base_color = (*np.random.uniform(0.1, 0.9, 3), 1.0)
        material.set_principled_shader_value("Base Color", base_color)
        material.set_principled_shader_value("Roughness", random.uniform(0.2, 1.0))
        material.set_principled_shader_value("Metallic", random.uniform(0.0, 0.8))
        distractor.add_material(material)

        distractor.set_cp("category_id", 2)
        distractors.append(distractor)

    return distractors


def apply_image_texture(material, image_path):
    """Apply an image texture to the material using object coordinates."""
    # Ensure material uses nodes
    mat = material.blender_obj
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear all nodes and start fresh
    nodes.clear()

    # Create output and BSDF
    output = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")

    # Create texture nodes
    coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    tex_image = nodes.new(type="ShaderNodeTexImage")

    # Load image
    try:
        tex_image.image = bpy.data.images.load(str(image_path))
    except:
        # If image already loaded, get it from data
        tex_image.image = bpy.data.images.get(
            Path(image_path).name
        ) or bpy.data.images.load(str(image_path))

    tex_image.interpolation = "Linear"

    # Use Object coordinates (better for cylindrical mugs than Generated)
    links.new(coord.outputs["Object"], mapping.inputs["Vector"])

    # Random scale and rotation for variety
    scale_factor = np.random.uniform(2.0, 5.0)
    mapping.inputs["Scale"].default_value = [scale_factor, scale_factor, scale_factor]
    mapping.inputs["Rotation"].default_value = [
        np.random.uniform(0, 2 * np.pi),
        np.random.uniform(0, 2 * np.pi),
        np.random.uniform(0, 2 * np.pi),
    ]

    # Connect nodes
    links.new(mapping.outputs["Vector"], tex_image.inputs["Vector"])
    links.new(tex_image.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Set material properties
    bsdf.inputs["Roughness"].default_value = random.uniform(0.3, 0.7)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.5


def create_procedural_texture(material, texture_type):
    """Create procedural texture patterns for mugs."""
    # Ensure material uses nodes
    mat = material.blender_obj
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear all nodes and start fresh
    nodes.clear()

    # Create output and BSDF
    output = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")

    # Base colors for the pattern
    color1 = (*np.random.uniform(0.2, 0.9, 3), 1.0)
    color2 = (*np.random.uniform(0.2, 0.9, 3), 1.0)

    if texture_type == "stripes":
        coord = nodes.new(type="ShaderNodeTexCoord")
        mapping = nodes.new(type="ShaderNodeMapping")
        wave = nodes.new(type="ShaderNodeTexWave")
        color_ramp = nodes.new(type="ShaderNodeValToRGB")

        # Random orientation
        rotation = [0, 0, 0]
        if np.random.random() > 0.5:
            rotation[0] = np.pi / 2
        else:
            rotation[2] = np.pi / 2
        mapping.inputs["Rotation"].default_value = rotation

        wave.inputs["Scale"].default_value = np.random.uniform(10, 30)
        wave.wave_type = "BANDS"

        links.new(coord.outputs["Object"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
        links.new(wave.outputs["Color"], color_ramp.inputs["Fac"])

        color_ramp.color_ramp.elements[0].color = color1
        color_ramp.color_ramp.elements[1].color = color2

        links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    elif texture_type == "dots":
        coord = nodes.new(type="ShaderNodeTexCoord")
        voronoi = nodes.new(type="ShaderNodeTexVoronoi")
        color_ramp = nodes.new(type="ShaderNodeValToRGB")

        voronoi.inputs["Scale"].default_value = np.random.uniform(15, 50)
        voronoi.feature = "DISTANCE_TO_EDGE"

        color_ramp.color_ramp.elements[0].position = np.random.uniform(0.3, 0.7)
        color_ramp.color_ramp.elements[0].color = color1
        color_ramp.color_ramp.elements[1].color = color2

        links.new(coord.outputs["Object"], voronoi.inputs["Vector"])
        links.new(voronoi.outputs["Distance"], color_ramp.inputs["Fac"])
        links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    elif texture_type == "noise":
        coord = nodes.new(type="ShaderNodeTexCoord")
        noise = nodes.new(type="ShaderNodeTexNoise")
        color_ramp = nodes.new(type="ShaderNodeValToRGB")

        noise.inputs["Scale"].default_value = np.random.uniform(3, 15)
        noise.inputs["Detail"].default_value = np.random.uniform(2, 6)

        color_ramp.color_ramp.elements[0].color = color1
        color_ramp.color_ramp.elements[1].color = color2

        links.new(coord.outputs["Object"], noise.inputs["Vector"])
        links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
        links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    elif texture_type == "checker":
        coord = nodes.new(type="ShaderNodeTexCoord")
        checker = nodes.new(type="ShaderNodeTexChecker")

        checker.inputs["Scale"].default_value = np.random.uniform(8, 25)
        checker.inputs["Color1"].default_value = color1
        checker.inputs["Color2"].default_value = color2

        links.new(coord.outputs["Object"], checker.inputs["Vector"])
        links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])

    # Connect BSDF to output
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Set material properties
    bsdf.inputs["Roughness"].default_value = random.uniform(0.3, 0.7)
    bsdf.inputs["Metallic"].default_value = 0.0


def load_texture_images(texture_dir):
    """Load all texture image paths from DTD directory structure."""
    texture_images = []
    texture_path = Path(texture_dir)

    if not texture_path.exists():
        return None

    # DTD has subdirectories for each texture category
    for subdir in texture_path.iterdir():
        if subdir.is_dir():
            texture_images.extend(subdir.glob("*.jpg"))
            texture_images.extend(subdir.glob("*.png"))

    return texture_images if texture_images else None


def validate_pose_data(rotation_matrix, translation_vector):
    """Validate pose data is physically valid."""
    try:
        R = np.array(rotation_matrix)
        t = np.array(translation_vector)

        # Check orthogonality
        should_be_identity = R @ R.T
        if not np.allclose(
            should_be_identity, np.eye(3), atol=ROTATION_ORTHOGONALITY_TOLERANCE
        ):
            return False

        # Check determinant
        if not np.allclose(np.linalg.det(R), 1.0, atol=ROTATION_DETERMINANT_TOLERANCE):
            return False

        # Check object is in front of camera (positive Z after NOCS coordinate transform)
        if t[2] <= 0:
            return False

        return True
    except Exception:
        return False


def save_sample_to_hdf5(data, output_dir):
    """
    Save sample data to HDF5 file.

    Args:
        data: Dictionary of data arrays to save
        output_dir: Directory path for output files
    """
    # BlenderProc always writes to numbered files (0.hdf5, 1.hdf5, etc.)
    # in the specified directory
    bproc.writer.write_hdf5(str(output_dir), data, append_to_existing_output=True)


# =======================================================================================
# MAIN GENERATION FUNCTION
# =======================================================================================


def generate_negative_sample(random_bg_path, K_matrix, image_width, image_height):
    """Generate a negative sample (background only, no object)."""
    # Dummy object for rendering pipeline
    dummy_obj = bproc.object.create_primitive("CUBE")
    dummy_obj.set_scale([0.001, 0.001, 0.001])
    dummy_obj.set_location([0, 0, -10])
    dummy_obj.hide(True)
    dummy_obj.set_cp("category_id", 0)

    # Camera setup
    camera_location = [0, -5, 3]
    rotation_matrix = bproc.camera.rotation_from_forward_vec(
        np.array([0, 0, 0]) - np.array(camera_location), inplane_rot=0
    )
    cam_pose = bproc.math.build_transformation_mat(camera_location, rotation_matrix)
    bproc.camera.add_camera_pose(cam_pose)
    bproc.camera.set_intrinsics_from_K_matrix(K_matrix, image_width, image_height)

    # Lighting
    light = bproc.types.Light()
    light.set_type("SUN")
    light.set_location([0, 0, 5])
    light.set_energy(1.0)

    # Compositor setup
    bproc.renderer.set_output_format(enable_transparency=True)
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree

    for node in tree.nodes:
        tree.nodes.remove(node)

    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    composite = tree.nodes.new(type="CompositorNodeComposite")
    alpha_over = tree.nodes.new(type="CompositorNodeAlphaOver")
    image_node = tree.nodes.new(type="CompositorNodeImage")
    scale_node = tree.nodes.new(type="CompositorNodeScale")

    image_node.image = bpy.data.images.load(random_bg_path)
    scale_node.space = "RENDER_SIZE"
    scale_node.frame_method = "CROP"

    tree.links.new(image_node.outputs["Image"], scale_node.inputs["Image"])
    tree.links.new(scale_node.outputs["Image"], alpha_over.inputs[1])
    tree.links.new(render_layers.outputs["Image"], alpha_over.inputs[2])
    tree.links.new(alpha_over.outputs["Image"], composite.inputs["Image"])

    bproc.renderer.enable_segmentation_output(map_by=["instance", "class"])

    # Render
    data = bproc.renderer.render()
    data["colors"] = [add_sensor_noise(data["colors"][0])]

    # Add empty labels
    data["nocs"] = [np.zeros((image_width, image_height, 3), dtype=np.float32)]
    data["bounding_box"] = [np.array([-1, -1, -1, -1], dtype=np.int32)]
    data["instance_mask"] = [np.zeros((image_width, image_height), dtype=np.uint8)]
    data["keypoint_heatmaps"] = [
        np.zeros((len(KEYPOINT_NAMES), HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    ]
    data["keypoints_2d"] = [
        np.full((len(KEYPOINT_NAMES), 2), -1, dtype=np.float32).tolist()
    ]
    data["keypoints_visibility"] = [
        np.zeros(len(KEYPOINT_NAMES), dtype=np.int32).tolist()
    ]
    data["keypoints_3d_canonical"] = [
        np.zeros((len(KEYPOINT_NAMES), 3), dtype=np.float32).tolist()
    ]
    data["model_name"] = [np.string_("NEGATIVE_SAMPLE")]
    data["is_negative"] = [np.array([1], dtype=np.uint8)]
    data["camera_intrinsics"] = [K_matrix.tolist()]
    data["camera_pose"] = [cam_pose.tolist()]
    data["object_to_camera_rotation"] = [np.eye(3).tolist()]
    data["object_to_camera_translation"] = [np.zeros(3).tolist()]
    data["metadata"] = [np.string_(json.dumps({"is_negative": True}))]

    dummy_obj.delete()

    return data


def main(args):
    """Main data generation pipeline."""
    print("=" * 70)
    print("Synthetic 6DoF Mug Pose Data Generator")
    print("=" * 70)

    # Initialize BlenderProc
    bproc.init()
    bproc.camera.set_resolution(args.width, args.height)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directories
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    if args.debug:
        Path(args.debug_dir).mkdir(exist_ok=True, parents=True)

    # Load assets
    print(f"\nLoading assets...")
    with open(args.keypoint_path, "r") as f:
        keypoints_db = json.load(f)

    annotated_models = [Path(args.model_dir) / name for name in keypoints_db.keys()]
    if not annotated_models:
        raise RuntimeError(f"No annotated models found in {args.keypoint_path}")

    coco_images = list(Path(args.coco_dir).glob("*.jpg"))
    if not coco_images:
        raise RuntimeError(f"No background images found in {args.coco_dir}")

    texture_images = load_texture_images(args.texture_dir)
    if texture_images:
        print(f"  ✓ {len(texture_images)} DTD texture images")
    else:
        raise RuntimeError(f"No textures found at {args.texture_dir}")

    print(f"  ✓ {len(annotated_models)} annotated models")
    print(f"  ✓ {len(coco_images)} background images")
    print(f"\nGenerating {args.num_samples} samples...")
    if args.debug:
        print(f"  Debug mode: ON (saving visualizations to {args.debug_dir})")
    print("=" * 70)

    successful_samples = 0

    with tqdm.tqdm(
        total=args.num_samples, desc="Generating samples", unit="sample"
    ) as pbar:
        while successful_samples < args.num_samples:
            # Decide if negative sample
            is_negative_sample = (
                args.generate_negatives and np.random.random() < NEGATIVE_SAMPLE_RATIO
            )

            # Clean up previous scene
            bproc.clean_up(clean_up_camera=True)

            # Select random background
            random_bg_path = str(np.random.choice(coco_images))

            # Generate camera intrinsics
            K_matrix = generate_camera_intrinsics(
                args.width, args.height, args, add_variation=True
            )

            # ===================================================================
            # NEGATIVE SAMPLE
            # ===================================================================

            if is_negative_sample:
                data = generate_negative_sample(
                    random_bg_path, K_matrix, args.width, args.height
                )
                save_sample_to_hdf5(data, args.output_dir)
                successful_samples += 1
                print("Generated negative sample")
                pbar.update(1)
                continue

            # ===================================================================
            # POSITIVE SAMPLE
            # ===================================================================

            obj_path = np.random.choice(annotated_models)

            objs = bproc.loader.load_obj(str(obj_path))
            if not objs:
                print(f"Failed to load obj from {obj_path}. Abandoning sample...")
                continue

            obj = objs[0]

            # Scale with variation
            scale_variation = np.random.uniform(*OBJECT_SCALE_VARIATION)
            uniform_scale = OBJECT_BASE_SCALE * scale_variation
            obj.set_scale([uniform_scale, uniform_scale, uniform_scale])
            obj.set_cp("category_id", 1)

            # Ambient light
            ambient = bproc.types.Light()
            ambient.set_type("SUN")
            ambient.set_location(AMBIENT_LIGHT_LOCATION)
            ambient.set_rotation_euler([0, 0, np.random.uniform(0, 2 * np.pi)])
            ambient.set_color(np.random.uniform(0.9, 1.0, 3))
            ambient.set_energy(np.random.uniform(*AMBIENT_ENERGY_RANGE))

            # Pose
            location = np.random.uniform(
                [
                    OBJECT_LOCATION_RANGE_X[0],
                    OBJECT_LOCATION_RANGE_Y[0],
                    OBJECT_LOCATION_RANGE_Z[0],
                ],
                [
                    OBJECT_LOCATION_RANGE_X[1],
                    OBJECT_LOCATION_RANGE_Y[1],
                    OBJECT_LOCATION_RANGE_Z[1],
                ],
            )
            obj.set_location(location)

            random_rotation = np.random.uniform(
                [
                    OBJECT_ROTATION_RANGE_X[0],
                    OBJECT_ROTATION_RANGE_Y[0],
                    OBJECT_ROTATION_RANGE_Z[0],
                ],
                [
                    OBJECT_ROTATION_RANGE_X[1],
                    OBJECT_ROTATION_RANGE_Y[1],
                    OBJECT_ROTATION_RANGE_Z[1],
                ],
            )
            obj.set_rotation_euler(random_rotation)

            # ===================================================================
            # Domain Randomization
            # ===================================================================

            # Material
            if len(obj.get_materials()) == 0:
                material = bproc.material.create(name="mug_material")
                obj.add_material(material)
            else:
                material = obj.get_materials()[0]

            material_type = np.random.choice(
                ["pbr", "solid", "patterned"], p=MATERIAL_TYPE_PROBS
            )

            if material_type == "pbr":
                base_color = (*np.random.uniform(*MATERIAL_COLOR_RANGE, 3), 1.0)
                material.set_principled_shader_value("Base Color", base_color)
                material.set_principled_shader_value(
                    "Roughness", random.uniform(*MATERIAL_ROUGHNESS_RANGE)
                )
                material.set_principled_shader_value(
                    "Metallic", random.uniform(*MATERIAL_METALLIC_RANGE)
                )
                material.set_principled_shader_value(
                    "Specular IOR Level", random.uniform(0.0, 1.0)
                )

                if np.random.random() < PROB_SUBSURFACE:
                    material.set_principled_shader_value(
                        "Subsurface Weight", random.uniform(0.0, 0.1)
                    )
                    material.set_principled_shader_value(
                        "Subsurface Radius", [0.1, 0.1, 0.1]
                    )

            elif material_type == "solid":
                base_color = (*np.random.uniform(*MATERIAL_COLOR_RANGE, 3), 1.0)
                material.set_principled_shader_value("Base Color", base_color)
                material.set_principled_shader_value("Roughness", 0.8)
                material.set_principled_shader_value("Metallic", 0.0)

            else:  # patterned
                # Try to use DTD texture images, fallback to procedural
                applied_texture = False

                if texture_images and np.random.random() < TEXTURE_APPLICATION_PROB:
                    try:
                        random_texture = np.random.choice(texture_images)
                        apply_image_texture(material, str(random_texture))
                        material_type = f"dtd_{random_texture.parent.name}"
                        applied_texture = True
                    except Exception as e:
                        # Fallback to procedural if texture loading fails
                        print(
                            f"Error when applying random texture from {random_texture}: {e}"
                        )
                        applied_texture = False

                if not applied_texture:
                    texture_type = np.random.choice(PROCEDURAL_TEXTURE_TYPES)
                    create_procedural_texture(material, texture_type)
                    material_type = f"procedural_{texture_type}"

            # Lighting
            num_lights = np.random.randint(*NUM_LIGHTS_RANGE)
            poi = obj.get_location()
            for _ in range(num_lights):
                light = bproc.types.Light()
                light.set_type(np.random.choice(LIGHT_TYPES))
                light_location = bproc.sampler.shell(
                    center=poi,
                    radius_min=LIGHT_RADIUS_RANGE[0],
                    radius_max=LIGHT_RADIUS_RANGE[1],
                )
                light.set_location(light_location)
                light.set_color(np.random.uniform(*LIGHT_COLOR_RANGE, 3))

                if light.get_type() == "SUN":
                    light.set_energy(np.random.uniform(*LIGHT_ENERGY_SUN))
                elif light.get_type() == "POINT":
                    light.set_energy(np.random.uniform(*LIGHT_ENERGY_POINT))
                else:  # SPOT
                    light.set_energy(np.random.uniform(*LIGHT_ENERGY_SPOT))
                # Point light at poi
                direction = poi - light_location
                rotation_matrix = bproc.camera.rotation_from_forward_vec(direction)
                euler = Matrix(rotation_matrix).to_euler()
                light.set_rotation_euler(euler)

            # Distractors
            distractors = []
            if np.random.random() < PROB_DISTRACTOR_OBJECTS:
                distractors = add_distractor_objects(location)

            # ===================================================================
            # Camera Placement
            # ===================================================================

            poi = obj.get_location()

            camera_location = bproc.sampler.shell(
                center=poi,
                radius_min=CAMERA_RADIUS_MIN,
                radius_max=CAMERA_RADIUS_MAX,
                elevation_min=CAMERA_ELEVATION_MIN,
                elevation_max=CAMERA_ELEVATION_MAX,
            )

            rotation_matrix = bproc.camera.rotation_from_forward_vec(
                poi - camera_location,
                inplane_rot=np.random.uniform(
                    -CAMERA_INPLANE_ROT_RANGE, CAMERA_INPLANE_ROT_RANGE
                ),
            )

            cam_pose = bproc.math.build_transformation_mat(
                camera_location, rotation_matrix
            )

            bproc.camera.add_camera_pose(cam_pose)
            bproc.camera.set_intrinsics_from_K_matrix(K_matrix, args.width, args.height)

            # Camera effects
            add_camera_effects(obj)

            # ===================================================================
            # Compute Pose (NOCS-compatible)
            # ===================================================================

            obj_to_cam_transform = compute_object_to_camera_transform(obj, cam_pose)
            obj_to_cam_rotation = obj_to_cam_transform[:3, :3]
            obj_to_cam_translation = obj_to_cam_transform[:3, 3]

            # CRITICAL: Transform to NOCS coordinate system
            obj_to_cam_rotation[1, :] *= -1
            obj_to_cam_rotation[2, :] *= -1

            # Re-orthogonalize
            U, _, Vt = np.linalg.svd(obj_to_cam_rotation)
            obj_to_cam_rotation = U @ Vt

            if np.linalg.det(obj_to_cam_rotation) < 0:
                U[:, -1] *= -1
                obj_to_cam_rotation = U @ Vt

            # Transform translation
            # obj_to_cam_translation = obj_to_cam_translation / uniform_scale
            obj_to_cam_translation[1] *= -1
            obj_to_cam_translation[2] *= -1

            # Validate
            if not validate_pose_data(obj_to_cam_rotation, obj_to_cam_translation):
                print("Failed to validate pose data. Abandoning sample...")
                continue

            # ===================================================================
            # Render
            # ===================================================================

            bproc.renderer.set_output_format(enable_transparency=True)

            # NOCS
            for distractor in distractors:
                distractor.hide(True)

            nocs_data = bproc.renderer.render_nocs(output_dir=None, return_data=True)

            for distractor in distractors:
                distractor.hide(False)

            # RGB with background compositor
            scene = bpy.context.scene
            scene.use_nodes = True
            tree = scene.node_tree

            for node in tree.nodes:
                tree.nodes.remove(node)

            render_layers = tree.nodes.new(type="CompositorNodeRLayers")
            composite = tree.nodes.new(type="CompositorNodeComposite")
            alpha_over = tree.nodes.new(type="CompositorNodeAlphaOver")
            image_node = tree.nodes.new(type="CompositorNodeImage")
            scale_node = tree.nodes.new(type="CompositorNodeScale")

            image_node.image = bpy.data.images.load(random_bg_path)
            scale_node.space = "RENDER_SIZE"
            scale_node.frame_method = "CROP"

            tree.links.new(image_node.outputs["Image"], scale_node.inputs["Image"])
            tree.links.new(scale_node.outputs["Image"], alpha_over.inputs[1])
            tree.links.new(render_layers.outputs["Image"], alpha_over.inputs[2])
            tree.links.new(alpha_over.outputs["Image"], composite.inputs["Image"])

            bproc.renderer.enable_segmentation_output(map_by=["instance", "class"])

            data = bproc.renderer.render()
            data["colors"] = [add_sensor_noise(data["colors"][0])]

            # Validate image isn't too dark
            rgb_image = data["colors"][0][:, :, :3]  # Remove alpha channel if present

            # Find pixels where ALL RGB channels are <= 10
            very_dark_pixels = np.all(rgb_image <= 10, axis=2)
            dark_pixel_count = np.sum(very_dark_pixels)
            total_pixels = args.width * args.height
            dark_pixel_fraction = dark_pixel_count / total_pixels

            # Discard if a good portion of image is very dark
            dark_pixel_fraction_threshold = 0.2
            if dark_pixel_fraction > dark_pixel_fraction_threshold:
                print(
                    f"At least {dark_pixel_fraction_threshold*100}% of image is very dark . Abandoning sample..."
                )
                continue

            # ===================================================================
            # Generate Labels
            # ===================================================================

            model_keypoints_3d = keypoints_db[obj_path.name]
            ordered_kps_3d = np.array(
                [model_keypoints_3d[name] for name in KEYPOINT_NAMES]
            )

            kps_2d_raw, kps_visibility = get_keypoints_and_visibility(
                obj,
                ordered_kps_3d,
                cam_pose,
                K_matrix,
                args.width,
                args.height,
                frame=0,
            )

            # Heatmaps
            keypoint_heatmaps = []
            for kp_2d, vis in zip(kps_2d_raw, kps_visibility):
                if vis == 2:
                    kp_heatmap = [
                        kp_2d[0] * HEATMAP_SIZE / args.width,
                        kp_2d[1] * HEATMAP_SIZE / args.height,
                    ]
                    heatmap = generate_gaussian_heatmap(
                        kp_heatmap, HEATMAP_SIZE, HEATMAP_SIGMA
                    )
                else:
                    heatmap = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
                keypoint_heatmaps.append(heatmap)

            keypoint_heatmaps = np.stack(keypoint_heatmaps, axis=0)

            # Mask and bbox
            class_segmap = data["class_segmaps"][0]
            mug_mask = (class_segmap == 1).astype(np.uint8)

            bbox = get_bounding_box_from_mask(mug_mask)
            if bbox is None:
                print("Failed to get bounding box. Abandoning sample...")
                continue

            # Metadata
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            bbox_area_fraction = bbox_area / (args.width * args.height)
            mask_area = np.sum(mug_mask > 0)
            occlusion_ratio = mask_area / bbox_area if bbox_area > 0 else 0.0
            is_truncated = (
                bbox[0] <= 0
                or bbox[1] <= 0
                or bbox[2] >= (args.width - 1)
                or bbox[3] >= (args.height - 1)
            )

            # Debug visualization
            if args.debug:
                save_debug_visualization(
                    data["colors"][0][:, :, :3],
                    kps_2d_raw,
                    kps_visibility,
                    bbox,
                    mug_mask,
                    nocs_data["nocs"][0],
                    keypoint_heatmaps,
                    successful_samples,
                    obj_path.name,
                )

            # ===================================================================
            # Package and Save
            # ===================================================================

            data.update(nocs_data)
            data["bounding_box"] = [bbox]
            data["instance_mask"] = [mug_mask]
            data["keypoint_heatmaps"] = [keypoint_heatmaps]
            data["keypoints_2d"] = [kps_2d_raw.tolist()]
            data["keypoints_visibility"] = [kps_visibility.tolist()]
            data["keypoints_3d_canonical"] = [ordered_kps_3d.tolist()]
            data["model_name"] = [np.string_(obj_path.name)]
            data["is_negative"] = [np.array([0], dtype=np.uint8)]
            data["camera_intrinsics"] = [K_matrix.tolist()]
            data["camera_pose"] = [cam_pose.tolist()]
            data["object_to_camera_rotation"] = [obj_to_cam_rotation.tolist()]
            data["object_to_camera_translation"] = [obj_to_cam_translation.tolist()]

            metadata = {
                "camera_distance": float(np.linalg.norm(cam_pose[:3, 3] - poi)),
                "object_distance_to_camera": float(
                    np.linalg.norm(obj_to_cam_translation)
                ),
                "num_lights": num_lights,
                "has_distractors": len(distractors) > 0,
                "num_distractors": len(distractors),
                "material_type": material_type,
                "object_scale": float(uniform_scale),
                "visible_keypoints": int(np.sum(kps_visibility == 2)),
                "occluded_keypoints": int(np.sum(kps_visibility == 1)),
                "out_of_frame_keypoints": int(np.sum(kps_visibility == 0)),
                "bbox_area_fraction": float(bbox_area_fraction),
                "occlusion_ratio": float(occlusion_ratio),
                "is_truncated": bool(is_truncated),
            }
            data["metadata"] = [np.string_(json.dumps(metadata))]

            save_sample_to_hdf5(data, args.output_dir)
            successful_samples += 1
            pbar.update(1)

        print("\n" + "=" * 70)
        print(f"✓ Generation complete!")
    print(f"  Samples generated: {args.num_samples}")
    print(f"  Output directory: {args.output_dir}")
    if args.debug:
        print(f"  Debug images: {args.debug_dir}")
    print("=" * 70)


# =======================================================================================
# CLI ARGUMENT PARSING
# =======================================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic 6DoF mug pose estimation training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Generation parameters
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help="Image width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
        help="Image height in pixels",
    )
    parser.add_argument(
        "--fx",
        type=float,
        default=None,
        help="Camera focal length X",
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=None,
        help="Camera focal length Y",
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=None,
        help="Camera principal point X",
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=None,
        help="Camera principal point Y",
    )

    # Paths
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing canonicalized mug models",
    )
    parser.add_argument(
        "--keypoint-path",
        type=str,
        default=str(DEFAULT_KEYPOINT_PATH),
        help="Path to canonical keypoints JSON file",
    )
    parser.add_argument(
        "--coco-dir",
        type=str,
        default=str(DEFAULT_COCO_DIR),
        help="Directory containing COCO background images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=str(DEFAULT_DEBUG_DIR),
        help="Directory for debug visualizations",
    )
    parser.add_argument(
        "--texture-dir",
        type=str,
        default=str(DEFAULT_TEXTURE_DIR),
        help="Directory containing DTD texture images",
    )

    # Options
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (save visualization images)",
    )
    parser.add_argument(
        "--no-negatives",
        dest="generate_negatives",
        action="store_false",
        help="Disable generation of negative samples",
    )

    args = parser.parse_args()

    main(args)
