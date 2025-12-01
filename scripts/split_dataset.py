import argparse
import random
import shutil
from pathlib import Path

from tqdm import tqdm


def process_subset(file_list, destination_dir, operation_name):
    """
    Copies files to destination_dir, renaming them sequentially
    """
    destination_path = Path(destination_dir)
    destination_path.mkdir(parents=True, exist_ok=True)

    total_files = len(file_list)
    if total_files == 0:
        return 0

    # Calculate padding based on THIS subset's size
    pad_width = len(str(total_files))

    print(f"  -> Writing {total_files} files to '{destination_dir}'...")

    for i, src_file in enumerate(tqdm(file_list, desc=operation_name, unit="file")):
        # Generate new sequential filename
        new_filename = f"{i:0{pad_width}d}.hdf5"
        dest_path = destination_path / new_filename

        shutil.copy2(src_file, dest_path)

    return total_files


def split_dataset(input_dir, output_dir, train_ratio, seed):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Scan and Sort
    print(f"Scanning '{input_dir}'...")
    all_files = sorted(list(input_path.glob("*.hdf5")))
    total_files = len(all_files)

    if total_files == 0:
        print("ERROR: No HDF5 files found in input directory.")
        return

    # Shuffle
    print(f"Found {total_files} files. Shuffling with seed {seed}...")
    random.seed(seed)
    random.shuffle(all_files)

    # Calculate Split
    split_idx = int(total_files * train_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print("Split Configuration:")
    print(
        f"  Train: {len(train_files)} files ({(len(train_files) / total_files) * 100:.1f}%)"
    )
    print(
        f"  Val:   {len(val_files)} files ({(len(val_files) / total_files) * 100:.1f}%)"
    )
    print("-" * 60)

    # Process Train
    process_subset(train_files, output_path / "train", "Processing Train")

    # Process Val
    process_subset(val_files, output_path / "val", "Processing Val")

    print("-" * 60)
    print("Split Complete.")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into Train/Val folders with re-indexing."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Source directory containing HDF5 files"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Destination directory (will create 'train' and 'val' subfolders)",
    )
    parser.add_argument(
        "-r",
        "--ratio",
        type=float,
        default=0.8,
        help="Ratio of files for Training (0.0 - 1.0)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling")
    args = parser.parse_args()

    if not (0 < args.ratio < 1):
        print("ERROR: Ratio must be between 0 and 1")
        exit(1)

    split_dataset(args.input, args.output, args.ratio, args.seed)
