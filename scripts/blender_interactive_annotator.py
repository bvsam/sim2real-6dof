# blender_interactive_annotator.py
"""
Interactive keypoint annotation tool for canonicalized mug models.

Usage:
1. Run this script in Blender's Scripting tab to register the tool
2. Find "Mug Tool" panel in 3D Viewport sidebar (press N)
3. Enter model filename and click "Load Model & Start"
4. Use Shift+Right-Click to place 3D cursor on keypoints
5. Click "Save [keypoint_name]" to record each point
6. Keypoints are saved to canonical_keypoints.json

Canonical coordinate system (as saved in OBJ file):
- Origin: Center of mug
- +Z: Up (opening direction)
- +X: Handle direction
- Scale: Fits in 1x1x1 cube

Note: A +90° X rotation is applied for visualization only.
Keypoints are saved in the OBJ's original coordinate system.
"""

bl_info = {
    "name": "Mug Keypoint Annotator",
    "blender": (3, 0, 0),
    "category": "Object",
}

import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

# --- Configuration ---
BASE_DIR = Path("/home/blender/workspace")

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


# --- Helper Functions ---


def get_center_of_extremity(context, mode="TOP"):
    """
    Calculate the center of the topmost or bottommost vertices.

    Args:
        mode: "TOP" for rim center, "BOTTOM" for base center

    Returns:
        3D world coordinate of the center point
    """
    obj = context.active_object
    if not obj or obj.type != "MESH":
        return None

    # Ensure we're in object mode
    if obj.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    # Get vertices in world space
    mesh = obj.data
    verts = np.array([v.co for v in mesh.vertices])
    matrix_world = np.array(obj.matrix_world)
    verts_4d = np.hstack([verts, np.ones((verts.shape[0], 1))])
    world_verts = (matrix_world @ verts_4d.T).T[:, :3]

    # Find extreme vertices along Z axis
    if mode == "TOP":
        z_extremity = world_verts[:, 2].max()
        extremity_mask = world_verts[:, 2] > (z_extremity - 0.01)
    else:  # BOTTOM
        z_extremity = world_verts[:, 2].min()
        extremity_mask = world_verts[:, 2] < (z_extremity + 0.01)

    extremity_verts = world_verts[extremity_mask]
    if extremity_verts.shape[0] == 0:
        return None

    return extremity_verts.mean(axis=0)


# --- Property Group ---


class MugAnnotatorProperties(bpy.types.PropertyGroup):
    model_to_annotate: bpy.props.StringProperty(
        name="Model Filename",
        default="cup_0008.obj",
        description="Filename of the model to annotate (e.g., cup_0008.obj)",
    )
    keypoint_index: bpy.props.IntProperty(name="Current Keypoint Index", default=0)
    is_active: bpy.props.BoolProperty(name="Is Annotation Active", default=False)
    stored_points_json: bpy.props.StringProperty(name="Stored Keypoints JSON")
    nudge_amount: bpy.props.FloatProperty(
        name="Nudge Step",
        default=0.005,
        min=0.0001,
        max=0.1,
        step=0.01,
        precision=4,
        description="Fine-tune cursor position increment",
    )


# --- Operators ---


class MUG_OT_nudge_cursor(bpy.types.Operator):
    """Fine-tune 3D cursor position"""

    bl_idname = "mug.nudge_cursor"
    bl_label = "Nudge 3D Cursor"

    axis: bpy.props.StringProperty()
    direction: bpy.props.FloatProperty()

    def execute(self, context):
        props = context.scene.mug_annotator_props
        amount = props.nudge_amount * self.direction

        if self.axis == "X":
            context.scene.cursor.location.x += amount
        elif self.axis == "Y":
            context.scene.cursor.location.y += amount
        elif self.axis == "Z":
            context.scene.cursor.location.z += amount

        return {"FINISHED"}


class MUG_OT_calculate_rim_center(bpy.types.Operator):
    """Automatically calculate and snap to rim center"""

    bl_idname = "mug.calculate_rim_center"
    bl_label = "Calculate Rim Center"

    def execute(self, context):
        center = get_center_of_extremity(context, mode="TOP")
        if center is not None:
            context.scene.cursor.location = center
            self.report({"INFO"}, "Cursor snapped to rim center")
        else:
            self.report({"ERROR"}, "Could not calculate rim center")
        return {"FINISHED"}


class MUG_OT_calculate_base_center(bpy.types.Operator):
    """Automatically calculate and snap to base center"""

    bl_idname = "mug.calculate_base_center"
    bl_label = "Calculate Base Center"

    def execute(self, context):
        center = get_center_of_extremity(context, mode="BOTTOM")
        if center is not None:
            context.scene.cursor.location = center
            self.report({"INFO"}, "Cursor snapped to base center")
        else:
            self.report({"ERROR"}, "Could not calculate base center")
        return {"FINISHED"}


class MUG_OT_setup_annotation(bpy.types.Operator):
    """Load model and prepare for annotation"""

    bl_idname = "mug.setup_annotation"
    bl_label = "Load Model & Start"

    def execute(self, context):
        props = context.scene.mug_annotator_props

        # Build model path
        model_path = (
            BASE_DIR / "data" / "ModelNet40_canonicalized" / props.model_to_annotate
        )

        if not model_path.exists():
            self.report({"ERROR"}, f"Model not found: {model_path}")
            return {"CANCELLED"}

        # Clear scene
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

        # Import OBJ
        bpy.ops.wm.obj_import(filepath=str(model_path))

        # Setup imported object
        obj = context.active_object
        if obj:
            # Apply +90° X rotation to display mug upright during annotation
            # (Keypoints are still saved in the OBJ's original coordinate system)
            obj.rotation_euler = (math.pi / 2, 0, 0)
            obj.location = (0, 0, 0)
            obj.scale = (1, 1, 1)

            context.view_layer.update()

            self.report({"INFO"}, f"✓ Loaded {props.model_to_annotate}")

        # Reset annotation state
        props.keypoint_index = 0
        props.stored_points_json = "{}"
        props.is_active = True

        return {"FINISHED"}


class MUG_OT_record_keypoint(bpy.types.Operator):
    """Record current cursor position as keypoint"""

    bl_idname = "mug.record_keypoint"
    bl_label = "Record Keypoint"

    def execute(self, context):
        props = context.scene.mug_annotator_props

        if not props.is_active:
            self.report({"ERROR"}, "Please load a model first")
            return {"CANCELLED"}

        obj = context.active_object
        if not obj:
            self.report({"ERROR"}, "No active object found")
            return {"CANCELLED"}

        # Get cursor position in world space
        cursor_world = context.scene.cursor.location.copy()

        # Transform to object's local space
        # This automatically accounts for the +90° rotation,
        # so keypoints are saved in the OBJ file's original coordinate system
        obj_matrix_world_inv = obj.matrix_world.inverted()
        cursor_local = obj_matrix_world_inv @ cursor_world

        # Store keypoint
        stored_points = json.loads(props.stored_points_json)
        current_name = KEYPOINT_NAMES[props.keypoint_index]
        stored_points[current_name] = list(cursor_local)

        self.report({"INFO"}, f"✓ Recorded '{current_name}'")
        props.stored_points_json = json.dumps(stored_points)
        props.keypoint_index += 1

        # Check if we're done
        if props.keypoint_index >= len(KEYPOINT_NAMES):
            self.save_to_file(context)
            props.is_active = False
        else:
            next_name = KEYPOINT_NAMES[props.keypoint_index]
            self.report({"INFO"}, f"Next: '{next_name}'")

        return {"FINISHED"}

    def save_to_file(self, context):
        """Save all keypoints to JSON file"""
        props = context.scene.mug_annotator_props
        json_path = BASE_DIR / "annotations" / "canonical_keypoints.json"

        # Load existing annotations
        all_data = {}
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    content = f.read()
                    if content:
                        all_data = json.loads(content)
            except json.JSONDecodeError:
                self.report({"WARNING"}, "Existing JSON corrupted, will overwrite")
                all_data = {}

        # Add new annotations
        newly_annotated_data = json.loads(props.stored_points_json)
        all_data[props.model_to_annotate] = newly_annotated_data

        # === VERIFICATION: Create spheres at saved keypoint positions ===
        obj = context.active_object
        for name, coords in newly_annotated_data.items():
            # Transform from object local space to world space
            # (accounting for the +90° rotation we applied)
            local_coord = Vector(coords)
            world_coord = obj.matrix_world @ local_coord

            # Create verification sphere in world space
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02, location=world_coord)
            sphere = context.active_object
            sphere.name = f"VERIFY_{name}"

            # Color it green for verification
            mat = bpy.data.materials.new(name=f"verify_mat_{name}")
            mat.diffuse_color = (0, 1, 0, 1)  # Green
            sphere.data.materials.append(mat)

        self.report(
            {"INFO"},
            "✓ VERIFICATION: Green spheres show where keypoints will be loaded",
        )
        # === END VERIFICATION ===

        # Save to file
        with open(json_path, "w") as f:
            json.dump(all_data, f, indent=4)

        self.report(
            {"INFO"},
            f"✓ Saved {len(newly_annotated_data)} keypoints for {props.model_to_annotate}",
        )


# --- UI Panel ---


class MUG_PT_annotator_panel(bpy.types.Panel):
    """Main annotation tool panel"""

    bl_label = "Mug Annotator"
    bl_idname = "OBJECT_PT_mug_annotator"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mug Tool"

    def draw(self, context):
        layout = self.layout
        props = context.scene.mug_annotator_props

        # Step 1: Load Model
        layout.label(text="1. Load Model:", icon="IMPORT")
        box = layout.box()
        box.prop(props, "model_to_annotate")
        box.operator("mug.setup_annotation", icon="PLAY")

        layout.separator()

        # Step 2: Annotate
        layout.label(text="2. Annotate Keypoints:", icon="TRACKER")
        box = layout.box()

        if props.is_active and props.keypoint_index < len(KEYPOINT_NAMES):
            current_name = KEYPOINT_NAMES[props.keypoint_index]

            box.label(text=f"Current: '{current_name}'", icon="CURSOR")

            # Auto-snap helpers for center keypoints
            if "center" in current_name:
                box.label(text="Quick Snap:", icon="SNAP_ON")
                row = box.row(align=True)
                if "rim" in current_name:
                    row.operator("mug.calculate_rim_center", text="→ Rim Center")
                elif "base" in current_name:
                    row.operator("mug.calculate_base_center", text="→ Base Center")

            box.separator()

            # Record button
            box.operator(
                "mug.record_keypoint", text=f"Save '{current_name}'", icon="CHECKMARK"
            )

            # Progress indicator
            progress = props.keypoint_index / len(KEYPOINT_NAMES)
            box.separator()
            box.label(
                text=f"Progress: {props.keypoint_index}/{len(KEYPOINT_NAMES)} ({progress*100:.0f}%)",
                icon="TIME",
            )

        elif props.is_active:
            box.label(text="✓ Annotation Complete!", icon="CHECKMARK")
            box.label(text="Load another model to continue.")
        else:
            box.label(text="Load a model to begin", icon="INFO")

        layout.separator()

        # Step 3: Fine-tune
        layout.label(text="3. Fine-Tune Cursor:", icon="PIVOT_CURSOR")
        box = layout.box()
        box.prop(props, "nudge_amount")

        col = box.column(align=True)

        # X axis
        row = col.row(align=True)
        op = row.operator("mug.nudge_cursor", text="-X")
        op.axis, op.direction = "X", -1.0
        row.label(text="X")
        op = row.operator("mug.nudge_cursor", text="+X")
        op.axis, op.direction = "X", 1.0

        # Y axis
        row = col.row(align=True)
        op = row.operator("mug.nudge_cursor", text="-Y")
        op.axis, op.direction = "Y", -1.0
        row.label(text="Y")
        op = row.operator("mug.nudge_cursor", text="+Y")
        op.axis, op.direction = "Y", 1.0

        # Z axis
        row = col.row(align=True)
        op = row.operator("mug.nudge_cursor", text="-Z")
        op.axis, op.direction = "Z", -1.0
        row.label(text="Z")
        op = row.operator("mug.nudge_cursor", text="+Z")
        op.axis, op.direction = "Z", 1.0


# --- Registration ---


classes = (
    MugAnnotatorProperties,
    MUG_OT_nudge_cursor,
    MUG_OT_calculate_rim_center,
    MUG_OT_calculate_base_center,
    MUG_OT_setup_annotation,
    MUG_OT_record_keypoint,
    MUG_PT_annotator_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mug_annotator_props = bpy.props.PointerProperty(
        type=MugAnnotatorProperties
    )


def unregister():
    del bpy.types.Scene.mug_annotator_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    # Unregister first to allow re-running the script
    try:
        unregister()
    except:
        pass
    register()
    print("✓ Mug Keypoint Annotator registered successfully")
