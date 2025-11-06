#!/usr/bin/env python3

import blenderproc as bproc  # isort: skip

"""
Synthetic 6DoF Mug Pose Estimation Data Generator

Generates training data for NOCS-based category-level 6DoF pose estimation.
Uses BlenderProc to render mugs with domain randomization, COCO backgrounds,
and generates ground truth: NOCS maps, masks, bboxes, and poses.
"""
import argparse
import json
import logging
import random
from pathlib import Path

import bpy
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import yaml
from mathutils import Matrix
from PIL import Image

logger = logging.getLogger(__name__)

# =======================================================================================
# HELPER FUNCTIONS
# =======================================================================================


def setup_logging(log_level):
    """Configure logging with timestamp and level."""
    # Create formatter
    formatter = logging.Formatter(
        "[%(levelname)s] [%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def save_debug_visualization(
    rgb_image,
    bbox,
    mask,
    nocs_map,
    sample_idx,
    model_name,
    debug_dir,
):
    """Save debug visualization with masks, etc."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[1, 1].axis("off")

    # RGB with bbox
    ax = axes[0, 0]
    ax.imshow(rgb_image)

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

    ax.set_title("RGB with BBox")
    ax.axis("off")

    # Instance mask
    ax = axes[0, 1]
    ax.imshow(mask, cmap="gray")
    ax.set_title("Instance Mask")
    ax.axis("off")

    # NOCS map
    ax = axes[1, 0]
    ax.imshow(nocs_map[:, :, :3])  # Only RGB channels
    ax.set_title("NOCS Map")
    ax.axis("off")

    plt.suptitle(f"Sample {sample_idx}: {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    debug_path = Path(debug_dir) / f"sample_{sample_idx:05d}.png"
    logger.info(f"Saving debug image to {debug_path}")
    plt.savefig(debug_path, dpi=120, bbox_inches="tight")
    plt.close()

    # Save the RGB image separately
    image = Image.fromarray(rgb_image)
    debug_rgb_image_path = Path(debug_dir) / f"sample_{sample_idx:05d}_rgb.png"
    logger.info(f"Saving debug RGB image to {debug_rgb_image_path}")
    image.save(debug_rgb_image_path)


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


def add_sensor_noise(rgb_image, sensor_noise_config):
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
    shot_noise_strength = np.random.uniform(*sensor_noise_config["shot_noise_range"])
    shot_noise = (
        np.random.normal(0, 1, rgb.shape) * np.sqrt(rgb + 0.01) * shot_noise_strength
    )
    rgb_noisy = rgb + shot_noise

    # Read noise
    read_noise_std = np.random.uniform(*sensor_noise_config["read_noise_range"])
    read_noise = np.random.normal(0, read_noise_std, rgb.shape)
    rgb_noisy = rgb_noisy + read_noise

    # Color shift
    if np.random.random() < sensor_noise_config["color_shift"]["probability"]:
        color_shift = np.random.uniform(
            *sensor_noise_config["color_shift"]["range"], size=(1, 1, 3)
        )
        rgb_noisy = rgb_noisy * color_shift

    rgb_noisy = np.clip(rgb_noisy, 0, 1)

    # Recombine with alpha
    if alpha is not None:
        img_noisy = np.concatenate([rgb_noisy, alpha], axis=2)
    else:
        img_noisy = rgb_noisy

    return (img_noisy * 255).astype(rgb_image.dtype)


def add_camera_effects(obj, motion_blur_config, depth_of_field_config):
    """Add motion blur and depth of field effects."""
    # Motion blur
    if np.random.random() < motion_blur_config["probability"]:
        bpy.context.scene.render.use_motion_blur = True
        bpy.context.scene.render.motion_blur_shutter = np.random.uniform(
            *motion_blur_config["shutter_range"]
        )
    else:
        bpy.context.scene.render.use_motion_blur = False

    # Depth of field
    cam = bpy.context.scene.camera
    if np.random.random() < depth_of_field_config["probability"]:
        poi = obj.get_location()
        cam_location = np.array(cam.location)
        distance_to_poi = np.linalg.norm(cam_location - poi)

        cam.data.dof.use_dof = True
        cam.data.dof.focus_distance = distance_to_poi
        cam.data.dof.aperture_fstop = np.random.uniform(
            *depth_of_field_config["aperture_range"]
        )
    else:
        cam.data.dof.use_dof = False


def generate_camera_intrinsics(base_intrinsics, variation_config=None):
    """Generate camera intrinsic matrix with optional variation, scaled to resolution."""
    # If intrinsics manually specified, use those
    fx = base_intrinsics["fx"]
    fy = base_intrinsics["fy"]
    cx = base_intrinsics["cx"]
    cy = base_intrinsics["cy"]

    if variation_config is not None:
        focal_scale = np.random.uniform(
            variation_config["focal_scale"]["min"],
            variation_config["focal_scale"]["max"],
        )
        principal_point_scale = np.random.uniform(
            -variation_config["principal_point"], variation_config["principal_point"]
        )
        fx *= focal_scale
        fy *= focal_scale
        cx += principal_point_scale
        cy += principal_point_scale

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    return K


def add_distractor_objects(mug_location, distractor_config):
    """
    Add random primitive objects as distractors in a shell around the mug.

    Args:
        mug_location: 3D position of the mug (numpy array or list)

    Returns:
        List of distractor objects
    """
    num_distractors = np.random.randint(*distractor_config["count_range"])
    distractors = []

    mug_pos = np.array(mug_location)

    for i in range(num_distractors):
        primitive_type = np.random.choice(["cube", "sphere", "cylinder", "cone"])

        if primitive_type == "cube":
            distractor = bproc.object.create_primitive("CUBE")
            scale = np.random.uniform(*distractor_config["scale_range"], 3)
        elif primitive_type == "sphere":
            distractor = bproc.object.create_primitive("SPHERE")
            scale = np.random.uniform(*distractor_config["scale_range"], 3)
        elif primitive_type == "cylinder":
            distractor = bproc.object.create_primitive("CYLINDER")
            scale_xy = np.random.uniform(
                distractor_config["scale_range"][0], distractor_config["scale_range"][1]
            )
            scale_z = np.random.uniform(
                distractor_config["scale_range"][0] * 2,
                distractor_config["scale_range"][1] * 3,
            )
            scale = [scale_xy, scale_xy, scale_z]
        else:  # cone
            distractor = bproc.object.create_primitive("CONE")
            scale = np.random.uniform(*distractor_config["scale_range"], 3)

        distractor.set_scale(scale)

        # Place in a shell around the mug
        # Sample random point on horizontal circle around mug
        radius = np.random.uniform(
            distractor_config["radius"]["min"], distractor_config["radius"]["max"]
        )
        angle = np.random.uniform(0, 2 * np.pi)

        # Horizontal offset from mug
        offset_x = radius * np.cos(angle)
        offset_y = radius * np.sin(angle)

        # Vertical offset (keep near table surface)
        offset_z = np.random.uniform(*distractor_config["height_offset_range"])

        distractor_location = mug_pos + np.array([offset_x, offset_y, offset_z])
        distractor.set_location(distractor_location)

        distractor.set_rotation_euler(
            np.random.uniform([0, 0, 0], [np.pi * 2, np.pi * 2, np.pi * 2])
        )

        # Random material
        material = bproc.material.create(name=f"distractor_mat_{i}")
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


def validate_pose_data(
    rotation_matrix,
    translation_vector,
    rotation_ortho_tolerance=1e-5,
    rotation_det_tolerance=1e-5,
):
    """Validate pose data is physically valid."""
    try:
        R = np.array(rotation_matrix)
        t = np.array(translation_vector)

        # Check orthogonality
        should_be_identity = R @ R.T
        if not np.allclose(
            should_be_identity, np.eye(3), atol=rotation_ortho_tolerance
        ):
            return False

        # Check determinant
        if not np.allclose(np.linalg.det(R), 1.0, atol=rotation_det_tolerance):
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


def calculate_camera_poi_offset(camera_distance, offset_fraction):
    """
    Calculate random offset for camera point-of-interest.

    This offset is applied perpendicular to the camera view direction
    to move where the camera is "looking", which shifts the object's
    position within the frame.

    Args:
        camera_distance: Distance from camera to object
        offset_fraction: Maximum offset as fraction of distance (0.2 ≈ central 80%)

    Returns:
        3D offset vector in world coordinates
    """
    max_offset = camera_distance * offset_fraction
    # Random offset in horizontal and vertical directions
    offset_x = np.random.uniform(-max_offset, max_offset)
    offset_y = np.random.uniform(-max_offset, max_offset)
    # Small Z offset for additional variation
    offset_z = np.random.uniform(-max_offset, max_offset)
    return np.array([offset_x, offset_y, offset_z])


def check_object_in_frame(bbox, image_width, image_height, min_visible_fraction=0.5):
    """
    Check if object bounding box is sufficiently visible in frame.

    Args:
        bbox: Bounding box [x_min, y_min, x_max, y_max] or None
        image_width: Image width in pixels
        image_height: Image height in pixels
        min_visible_fraction: Minimum fraction of bbox that should be visible (0.5 = 50%)

    Returns:
        (is_valid, visible_fraction) tuple
    """
    if bbox is None or bbox[0] < 0:
        return False, 0.0

    # Calculate original bbox dimensions
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    bbox_area = bbox_width * bbox_height

    if bbox_area <= 0:
        return False, 0.0

    # Calculate clipped bbox (what's actually visible within image bounds)
    clipped_x_min = max(0, bbox[0])
    clipped_y_min = max(0, bbox[1])
    clipped_x_max = min(image_width, bbox[2])
    clipped_y_max = min(image_height, bbox[3])

    clipped_width = clipped_x_max - clipped_x_min
    clipped_height = clipped_y_max - clipped_y_min
    clipped_area = max(0, clipped_width * clipped_height)

    # Calculate what fraction of the bbox is visible
    visible_fraction = clipped_area / bbox_area if bbox_area > 0 else 0.0

    is_valid = visible_fraction >= min_visible_fraction

    return is_valid, visible_fraction


# =======================================================================================
# MAIN GENERATION FUNCTION
# =======================================================================================


def generate_negative_sample(
    random_bg_path, K_matrix, image_width, image_height, config
):
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
    data["colors"] = [
        add_sensor_noise(
            data["colors"][0], config["domain_randomization"]["sensor_noise"]
        )
    ]

    # Add empty labels
    data["nocs"] = [np.zeros((image_width, image_height, 3), dtype=np.float32)]
    data["bounding_box"] = [np.array([-1, -1, -1, -1], dtype=np.int32)]
    data["instance_mask"] = [np.zeros((image_width, image_height), dtype=np.uint8)]
    data["model_name"] = [np.string_("NEGATIVE_SAMPLE")]
    data["is_negative"] = [np.array([1], dtype=np.uint8)]
    data["camera_intrinsics"] = [K_matrix.tolist()]
    data["camera_pose"] = [cam_pose.tolist()]
    data["object_to_camera_rotation"] = [np.eye(3).tolist()]
    data["object_to_camera_translation"] = [np.zeros(3).tolist()]
    data["metadata"] = [np.string_(json.dumps({"is_negative": True}))]

    dummy_obj.delete()

    return data


def main():
    """Main data generation pipeline."""
    # =======================================================================================
    # CLI ARGUMENT PARSING
    # =======================================================================================
    parser = argparse.ArgumentParser(
        description="Generate synthetic 6DoF mug pose estimation training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    # Generation parameters
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        help="Number of synthetic samples to generate",
    )
    # Options
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (save visualization images)",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    parser.add_argument(
        "--no-negatives",
        dest="generate_negatives",
        action="store_false",
        help="Disable generation of negative samples",
    )
    args = parser.parse_args()

    logger = setup_logging(args.log_level)

    logger.info("=" * 70)
    logger.info("Synthetic 6DoF Mug Pose Data Generator")
    logger.info("=" * 70)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    IMAGE_WIDTH = config["generation"]["image_resolution"]["width"]
    IMAGE_HEIGHT = config["generation"]["image_resolution"]["height"]
    OUTPUT_DIR = config["paths"]["output_dir"]
    DEBUG_DIR = config["paths"]["debug_dir"]
    NUM_SAMPLES = (
        args.num_samples
        if args.num_samples is not None
        else config["generation"]["num_samples"]
    )
    SEED = args.seed if args.seed is not None else config["generation"]["seed"]

    # Initialize BlenderProc
    bproc.init()
    bproc.camera.set_resolution(IMAGE_WIDTH, IMAGE_HEIGHT)

    # Set random seed
    random.seed(SEED)
    np.random.seed(SEED)

    # Create output directories
    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    if args.debug:
        Path(DEBUG_DIR).mkdir(exist_ok=True, parents=True)

    # Load assets
    logger.info(f"Loading assets...")

    models = list(Path(config["paths"]["model_dir"]).glob("*.obj"))
    if not models:
        raise RuntimeError(
            f"No annotated models found in {config['paths']['model_dir']}"
        )

    coco_images = list(Path(config["paths"]["coco_dir"]).glob("*.jpg"))
    if not coco_images:
        raise RuntimeError(
            f"No background images found in {config['paths']['coco_dir']}"
        )

    texture_images = load_texture_images(config["paths"]["texture_dir"])
    if texture_images:
        logger.info(f"  ✓ {len(texture_images)} DTD texture images")
    else:
        raise RuntimeError(f"No textures found at {config['paths']['texture_dir']}")

    logger.info(f"  ✓ {len(models)} models")
    logger.info(f"  ✓ {len(coco_images)} background images")
    logger.info(f"Generating {NUM_SAMPLES} samples...")
    if args.debug:
        logger.info(f"  Debug mode: ON (saving visualizations to {DEBUG_DIR})")
    logger.info("=" * 70)

    successful_samples = 0

    with tqdm.tqdm(total=NUM_SAMPLES, desc="Generating samples", unit="sample") as pbar:
        while successful_samples < NUM_SAMPLES:
            # Decide if negative sample
            is_negative_sample = (
                args.generate_negatives
                and np.random.random() < config["generation"]["negative_sample_ratio"]
            )

            # Clean up previous scene
            bproc.clean_up(clean_up_camera=True)

            # Select random background
            random_bg_path = str(np.random.choice(coco_images))

            # Generate camera intrinsics
            K_matrix = generate_camera_intrinsics(
                config["camera"]["intrinsics"]["base"],
                variation_config=config["camera"]["intrinsics"]["variation"],
            )

            # ===================================================================
            # NEGATIVE SAMPLE
            # ===================================================================

            if is_negative_sample:
                data = generate_negative_sample(
                    random_bg_path, K_matrix, IMAGE_WIDTH, IMAGE_HEIGHT, config
                )
                save_sample_to_hdf5(data, OUTPUT_DIR)
                successful_samples += 1
                logger.info("Generated negative sample")
                pbar.update(1)
                continue

            # ===================================================================
            # POSITIVE SAMPLE
            # ===================================================================

            obj_path = np.random.choice(models)

            objs = bproc.loader.load_obj(str(obj_path))
            if not objs:
                logger.warning(
                    f"Failed to load obj from {obj_path}. Abandoning sample..."
                )
                continue

            obj = objs[0]

            # Scale with variation
            scale_variation = np.random.uniform(
                config["object"]["scale_variation"]["min"],
                config["object"]["scale_variation"]["max"],
            )
            # Re-normalize using canoncalized scale to ensure unit
            # size, then apply scaling as usual
            uniform_scale = (
                (1 / config["object"]["canonicalization_scale"])
                * config["object"]["base_scale"]
                * scale_variation
            )
            obj.set_scale([uniform_scale, uniform_scale, uniform_scale])
            obj.set_cp("category_id", 1)

            # Ambient light
            ambient = bproc.types.Light()
            ambient.set_type("SUN")
            ambient.set_location(
                config["domain_randomization"]["lighting"]["ambient"]["location"]
            )
            ambient.set_rotation_euler([0, 0, np.random.uniform(0, 2 * np.pi)])
            ambient.set_color(np.random.uniform(0.9, 1.0, 3))
            ambient.set_energy(
                np.random.uniform(
                    *config["domain_randomization"]["lighting"]["ambient"][
                        "energy_range"
                    ]
                )
            )

            # Pose
            location = np.random.uniform(
                [
                    config["object"]["location_range"]["x"][0],
                    config["object"]["location_range"]["y"][0],
                    config["object"]["location_range"]["z"][0],
                ],
                [
                    config["object"]["location_range"]["x"][1],
                    config["object"]["location_range"]["y"][1],
                    config["object"]["location_range"]["z"][1],
                ],
            )
            obj.set_location(location)

            random_rotation = np.random.uniform(
                [
                    config["object"]["rotation_range_rad"]["x"][0],
                    config["object"]["rotation_range_rad"]["y"][0],
                    config["object"]["rotation_range_rad"]["z"][0],
                ],
                [
                    config["object"]["rotation_range_rad"]["x"][1],
                    config["object"]["rotation_range_rad"]["y"][1],
                    config["object"]["rotation_range_rad"]["z"][1],
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
                ["pbr", "solid", "patterned"],
                p=config["domain_randomization"]["materials"]["type_probabilities"],
            )

            if material_type == "pbr":
                base_color = (
                    *np.random.uniform(
                        *config["domain_randomization"]["materials"]["color_range"], 3
                    ),
                    1.0,
                )
                material.set_principled_shader_value("Base Color", base_color)
                material.set_principled_shader_value(
                    "Roughness",
                    random.uniform(
                        *config["domain_randomization"]["materials"]["roughness_range"]
                    ),
                )
                material.set_principled_shader_value(
                    "Metallic",
                    random.uniform(
                        *config["domain_randomization"]["materials"]["metallic_range"]
                    ),
                )
                material.set_principled_shader_value(
                    "Specular IOR Level", random.uniform(0.0, 1.0)
                )

                if (
                    np.random.random()
                    < config["domain_randomization"]["subsurface_probability"]
                ):
                    material.set_principled_shader_value(
                        "Subsurface Weight", random.uniform(0.0, 0.1)
                    )
                    material.set_principled_shader_value(
                        "Subsurface Radius", [0.1, 0.1, 0.1]
                    )

            elif material_type == "solid":
                base_color = (
                    *np.random.uniform(
                        *config["domain_randomization"]["materials"]["color_range"], 3
                    ),
                    1.0,
                )
                material.set_principled_shader_value("Base Color", base_color)
                material.set_principled_shader_value("Roughness", 0.8)
                material.set_principled_shader_value("Metallic", 0.0)

            else:  # patterned
                # Try to use DTD texture images, fallback to procedural
                applied_texture = False

                if (
                    texture_images
                    and np.random.random()
                    < config["domain_randomization"]["materials"]["textures"][
                        "application_prob"
                    ]
                ):
                    try:
                        random_texture = np.random.choice(texture_images)
                        apply_image_texture(material, str(random_texture))
                        material_type = f"dtd_{random_texture.parent.name}"
                        applied_texture = True
                    except Exception as e:
                        # Fallback to procedural if texture loading fails
                        logger.warning(
                            f"Error when applying random texture from {random_texture}: {e}"
                        )
                        applied_texture = False

                if not applied_texture:
                    texture_type = np.random.choice(
                        config["domain_randomization"]["materials"]["textures"][
                            "procedural_types"
                        ]
                    )
                    create_procedural_texture(material, texture_type)
                    material_type = f"procedural_{texture_type}"

            # Lighting
            num_lights = np.random.randint(
                *config["domain_randomization"]["lighting"]["count_range"]
            )
            poi = obj.get_location()
            for _ in range(num_lights):
                light = bproc.types.Light()
                light.set_type(
                    np.random.choice(
                        config["domain_randomization"]["lighting"]["types"]
                    )
                )
                light_location = bproc.sampler.shell(
                    center=poi,
                    radius_min=config["domain_randomization"]["lighting"][
                        "radius_range"
                    ][0],
                    radius_max=config["domain_randomization"]["lighting"][
                        "radius_range"
                    ][1],
                )
                light.set_location(light_location)
                light.set_color(
                    np.random.uniform(
                        *config["domain_randomization"]["lighting"]["color_range"], 3
                    )
                )

                if light.get_type() == "SUN":
                    light.set_energy(
                        np.random.uniform(
                            *config["domain_randomization"]["lighting"]["energy"]["sun"]
                        )
                    )
                elif light.get_type() == "POINT":
                    light.set_energy(
                        np.random.uniform(
                            *config["domain_randomization"]["lighting"]["energy"][
                                "point"
                            ]
                        )
                    )
                else:  # SPOT
                    light.set_energy(
                        np.random.uniform(
                            *config["domain_randomization"]["lighting"]["energy"][
                                "spot"
                            ]
                        )
                    )
                # Point light at poi
                direction = poi - light_location
                rotation_matrix = bproc.camera.rotation_from_forward_vec(direction)
                euler = Matrix(rotation_matrix).to_euler()
                light.set_rotation_euler(euler)

            # Distractors
            distractors = []
            if (
                np.random.random()
                < config["domain_randomization"]["distractors"]["probability"]
            ):
                distractors = add_distractor_objects(
                    location, config["domain_randomization"]["distractors"]
                )

            # ===================================================================
            # Camera Placement (with offset for varied positioning)
            # ===================================================================

            poi = obj.get_location()

            # Calculate camera distance and offset for varied object position in frame
            camera_distance = np.random.uniform(
                config["camera"]["radius"]["min"], config["camera"]["radius"]["max"]
            )
            camera_location = bproc.sampler.shell(
                center=poi,
                radius_min=camera_distance,
                radius_max=camera_distance,
                elevation_min=config["camera"]["elevation"]["min"],
                elevation_max=config["camera"]["elevation"]["max"],
            )

            poi_offset = calculate_camera_poi_offset(
                camera_distance, config["camera"]["poi_offset_fraction"]
            )
            poi_with_offset = poi + poi_offset
            # Point camera at offset POI
            # This creates the off-center framing effect
            rotation_matrix = bproc.camera.rotation_from_forward_vec(
                poi_with_offset - camera_location,
                inplane_rot=np.random.uniform(
                    -config["camera"]["inplane_rotation_range"],
                    config["camera"]["inplane_rotation_range"],
                ),
            )

            cam_pose = bproc.math.build_transformation_mat(
                camera_location, rotation_matrix
            )

            bproc.camera.add_camera_pose(cam_pose)
            bproc.camera.set_intrinsics_from_K_matrix(
                K_matrix, IMAGE_WIDTH, IMAGE_HEIGHT
            )

            # Camera effects
            add_camera_effects(
                obj,
                config["domain_randomization"]["motion_blur"],
                config["domain_randomization"]["depth_of_field"],
            )

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
                logger.warning("Failed to validate pose data. Abandoning sample...")
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
            data["colors"] = [
                add_sensor_noise(
                    data["colors"][0], config["domain_randomization"]["sensor_noise"]
                )
            ]

            # Validate image isn't too dark
            rgb_image = data["colors"][0][:, :, :3]  # Remove alpha channel if present

            # Find pixels where ALL RGB channels are <= 10
            very_dark_pixels = np.all(rgb_image <= 10, axis=2)
            dark_pixel_count = np.sum(very_dark_pixels)
            total_pixels = IMAGE_WIDTH * IMAGE_HEIGHT
            dark_pixel_fraction = dark_pixel_count / total_pixels

            # Discard if a good portion of image is very dark
            dark_pixel_fraction_threshold = 0.2
            if dark_pixel_fraction > dark_pixel_fraction_threshold:
                logger.warning(
                    f"At least {dark_pixel_fraction_threshold*100}% of image is very dark . Abandoning sample..."
                )
                continue

            # ===================================================================
            # Generate Labels
            # ===================================================================

            # Mask and bbox
            class_segmap = data["class_segmaps"][0]
            mug_mask = (class_segmap == 1).astype(np.uint8)

            bbox = get_bounding_box_from_mask(mug_mask)
            if bbox is None:
                logger.warning("Failed to get bounding box. Abandoning sample...")
                continue

            # Check if object is sufficiently in frame
            is_in_frame, visible_fraction = check_object_in_frame(
                bbox,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
                config["validation"]["min_bbox_visible_fraction"],
            )

            if not is_in_frame:
                logger.warning(
                    f"Object not sufficiently in frame (only {visible_fraction*100:.1f}% of bbox visible). "
                    f"Abandoning sample..."
                )
                continue

            # Discard if large portion of object is occluded
            nocs = nocs_data["nocs"][0]
            # Count pixels where NOCS is non-zero (object present)
            nocs_mask = np.any(nocs[:, :, :3] > 0, axis=2)
            nocs_pixels_count = np.count_nonzero(nocs_mask)
            if nocs_pixels_count == 0:
                logger.warning(
                    f"Found 0 non-black NOCS pixels in output. Abandoning sample..."
                )
                continue
            coverage_score = np.sum(mug_mask) / nocs_pixels_count
            if coverage_score < (1 - config["validation"]["max_occlusion_allowable"]):
                logger.warning(
                    f"Object is {(1-coverage_score)*100}% occluded. Abandoning sample..."
                )
                continue

            # Metadata
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            bbox_area_fraction = bbox_area / (IMAGE_WIDTH * IMAGE_HEIGHT)
            mask_area = np.sum(mug_mask > 0)
            occlusion_ratio = mask_area / bbox_area if bbox_area > 0 else 0.0
            is_truncated = (
                bbox[0] <= 0
                or bbox[1] <= 0
                or bbox[2] >= (IMAGE_WIDTH - 1)
                or bbox[3] >= (IMAGE_HEIGHT - 1)
            )

            # Debug visualization
            if args.debug:
                save_debug_visualization(
                    data["colors"][0][:, :, :3],
                    bbox,
                    mug_mask,
                    nocs_data["nocs"][0],
                    successful_samples,
                    obj_path.name,
                    DEBUG_DIR,
                )

            # ===================================================================
            # Package and Save
            # ===================================================================

            data.update(nocs_data)
            data["bounding_box"] = [bbox]
            data["instance_mask"] = [mug_mask]
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
                "bbox_area_fraction": float(bbox_area_fraction),
                "occlusion_ratio": float(occlusion_ratio),
                "is_truncated": bool(is_truncated),
            }
            data["metadata"] = [np.string_(json.dumps(metadata))]

            save_sample_to_hdf5(data, OUTPUT_DIR)
            successful_samples += 1
            pbar.update(1)

    logger.info("=" * 70)
    logger.info(f"✓ Generation complete!")
    logger.info(f"  Samples generated: {NUM_SAMPLES}")
    logger.info(f"  Output directory: {OUTPUT_DIR}")
    if args.debug:
        logger.info(f"  Debug images: {DEBUG_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
