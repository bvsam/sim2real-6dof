import argparse
import concurrent.futures
import math
import tarfile
from pathlib import Path
from typing import List

from tqdm import tqdm


def get_file_number(file_path: Path) -> int:
    """
    Extracts the integer from a filename like '100.hdf5' -> 100
    """
    return int(file_path.stem)


def create_archive_worker(files: List[Path], output_path: Path, use_gzip: bool):
    """
    Worker function to create a single archive.
    Now accepts the specific output path directly.
    """
    if not files:
        return

    # Set mode based on gzip flag
    mode = "w:gz" if use_gzip else "w"

    # Create tar
    with tarfile.open(output_path, mode=mode) as tar:
        for file_path in files:
            # arcname=file_path.name ensures the tar is flat (no parent dirs)
            tar.add(file_path, arcname=file_path.name)

    return output_path.name


def main():
    parser = argparse.ArgumentParser(
        description="Archive HDF5 files sequentially in parallel."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="List of input directory paths",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where tar files will be stored",
    )
    parser.add_argument(
        "--shard-size", type=int, default=1000, help="Number of files per archive"
    )
    parser.add_argument(
        "--gzip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to gzip tar files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel processes",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    if not args.output.exists():
        print(f"Creating output directory: {args.output}")
        args.output.mkdir(parents=True, exist_ok=True)

    # Collect all shards
    # We gather every chunk from every directory into a single list first
    all_chunks = []

    print("Scanning directories...")
    for src_dir in args.inputs:
        if not src_dir.exists():
            print(f"Directory: {src_dir} not found. Skipping...")
            continue

        files = list(src_dir.glob("*.hdf5"))
        try:
            files.sort(key=get_file_number)
        except ValueError as e:
            print(f"Error sorting files in {src_dir}: {e}")
            continue

        # Create chunks and add to master list
        for i in range(0, len(files), args.shard_size):
            shard = files[i : i + args.shard_size]
            all_chunks.append((src_dir.name, shard))

    total_shards = len(all_chunks)
    if total_shards == 0:
        print("No files found.")
        return

    # Calculate padding
    pad_width = int(math.log10(total_shards)) + 1

    print(f"Found {total_shards} shards total. Using {pad_width}-digit zero padding.")
    print(f"Starting execution with {args.workers} parallel workers...")

    # Prepare tasks
    tasks = []
    ext = ".tar.gz" if args.gzip else ".tar"
    for i, (src_dir, shard) in enumerate(all_chunks):
        # Generate name: output_00.tar.gz
        filename = f"{src_dir}_{str(i).zfill(pad_width)}{ext}"
        out_path = args.output / filename
        tasks.append((shard, out_path, args.gzip))

    # Execute in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(create_archive_worker, shard, out_path, gz)
            for (shard, out_path, gz) in tasks
        ]
        pbar = tqdm(
            concurrent.futures.as_completed(futures), total=total_shards, unit="shard"
        )
        for future in pbar:
            try:
                result_name = future.result()
                pbar.set_description(f"Finished {result_name}")
            except Exception as e:
                print(f"An error occurred in a worker: {e}")

    print("All processing complete.")


if __name__ == "__main__":
    main()
