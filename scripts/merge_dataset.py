import argparse
import hashlib
import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def get_semantic_hash(hdf5_path):
    """
    Generates a hash based on the SCENE CONTENT (Pose + Model),
    ignoring pixel rendering noise.
    """
    with h5py.File(hdf5_path, "r") as f:
        # We create a signature from the things that define the scene physically
        model_name = str(np.array(f["model_name"]))

        # Round pose to 4 decimals to avoid tiny float point drift,
        # but keep it precise enough to distinguish samples.
        pose_trans = np.round(np.array(f["object_to_camera_translation"]), 4)
        pose_rot = np.round(np.array(f["object_to_camera_rotation"]), 4)

        # Combine into bytes
        signature = model_name.encode() + pose_trans.tobytes() + pose_rot.tobytes()

        return hashlib.md5(signature).hexdigest()


def merge_datasets(source_dirs, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Gather all source files
    all_files = []
    print("Scanning input directories...")
    for s_dir in source_dirs:
        path = Path(s_dir)
        if path.exists():
            # Sort locally to ensure deterministic order from each folder
            files = list(path.glob("*.hdf5")).sort(key=lambda x: int(x.stem))
            print(f"Found {len(files)} files in '{s_dir}'")
            all_files.extend(files)
        else:
            print(f"WARNING: Source directory '{s_dir}' does not exist.")

    total_files = len(all_files)
    if total_files == 0:
        print("No files found. Exiting.")
        return

    # Calculate Zero-Padding Width
    pad_width = len(str(total_files))
    print(
        f"Detected {total_files} potential files. Using {pad_width}-digit zero padding."
    )

    global_counter = 0
    seen_hashes = set()
    duplicates_dropped = 0
    corrupt_files = 0

    print("Merging and re-indexing...")
    for src_file in tqdm(all_files, unit="sample"):
        try:
            # Check for duplicates using Semantic Hashing
            semantic_hash = get_semantic_hash(src_file)

            if semantic_hash in seen_hashes:
                duplicates_dropped += 1
                continue

            seen_hashes.add(semantic_hash)

            # Generate new filename with zero padding
            new_filename = f"{global_counter:0{pad_width}d}.hdf5"
            dest_path = output_path / new_filename

            shutil.copy2(src_file, dest_path)

            global_counter += 1

        except Exception as e:
            print(f"ERROR reading {src_file}: {e}")
            corrupt_files += 1

    print("=" * 60)
    print("Merge Complete.")
    print(f"Total Unique Samples: {global_counter}")
    print(f"Duplicates Dropped: {duplicates_dropped}")
    print(f"Corrupt/Skipped: {corrupt_files}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple BlenderProc HDF5 output folders into one sequential dataset."
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        required=True,
        help="List of input directories",
    )
    parser.add_argument("-o", "--output", required=True, help="Final output directory")
    args = parser.parse_args()
    merge_datasets(args.inputs, args.output)


if __name__ == "__main__":
    main()
