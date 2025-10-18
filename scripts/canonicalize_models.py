# canonicalize_models.py
"""
Canonicalize mug models to a standard pose:
- Centered at origin
- Scaled to fit in 1x1x1 cube
- Upright orientation (opening on +Z, handle on +X)
"""

from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = Path("/home/blender/workspace")
data_root = BASE_DIR / "data"
approved_list_file = BASE_DIR / "annotations" / "approved_mugs_with_handles.txt"
output_dir = data_root / "ModelNet40_canonicalized"


def main():
    """
    Loads models from approved list, canonicalizes them, and saves as .obj files.

    Canonical pose:
    - Origin: Center of bounding box
    - Scale: Fits in 1x1x1 cube
    - Orientation: Upright (opening +Z, handle +X)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load approved model list
    off_files = []
    if not approved_list_file.is_file():
        print(f"Error: Approved list not found: {approved_list_file}")
        return

    print(f"Loading models from: {approved_list_file}")
    with approved_list_file.open("r") as f:
        for line in f:
            clean_line = line.strip()
            if clean_line:
                file_path = data_root / clean_line
                if file_path.exists():
                    off_files.append(file_path)
                else:
                    print(f"  [Warning] File not found: {file_path}")

    if not off_files:
        print("Error: No valid model files found.")
        return

    print(f"Processing {len(off_files)} models...\n")

    # Process each model
    for off_path in tqdm(off_files, desc="Canonicalizing"):
        try:
            # Load mesh
            mesh = trimesh.load(off_path, force="mesh")

            # Center at origin
            mesh.apply_translation(-mesh.bounding_box.centroid)

            # Scale to 1x1x1 cube
            mesh.apply_scale(1.0 / np.max(mesh.extents))

            # Rotate to upright pose (ModelNet40 mugs are typically on their side)
            # -90° around X makes opening point up (+Z) and handle point along +X
            rotation_matrix = trimesh.transformations.rotation_matrix(
                -np.pi / 2, [1, 0, 0]
            )
            mesh.apply_transform(rotation_matrix)

            # Save as OBJ
            output_path = output_dir / (off_path.stem + ".obj")
            mesh.export(output_path)

        except Exception as e:
            print(f"\nError processing {off_path.name}: {e}")

    print(f"\n✓ Canonicalization complete!")
    print(f"  Processed: {len(off_files)} models")
    print(f"  Saved to: {output_dir}")


if __name__ == "__main__":
    main()
