import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def download_file(url: str, destination: Path):
    print(f"Downloading from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024  # 1 MB

        progress_bar = tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            desc=destination.name,
        )

        with open(destination, "wb") as file:
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))
                file.write(chunk)
        progress_bar.close()

        if total_size != 0 and progress_bar.n != total_size:
            print("ERROR: Download incomplete.")
            destination.unlink(missing_ok=True)
            sys.exit(1)

        print("Download complete.")

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Download failed: {e}")
        destination.unlink(missing_ok=True)
        sys.exit(1)


def extract_archive(archive_path: Path, extract_to: Path):
    print(f"Extracting '{archive_path}'...")

    # Determine archive type by extension
    is_zip = archive_path.suffix == ".zip"
    is_tar = ".tar" in archive_path.name  # Handles .tar.gz, .tgz, .tar

    try:
        if is_zip:
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        elif is_tar:
            # "r:*" lets python determine if it's gz, bz2 or uncompressed
            with tarfile.open(archive_path, "r:*") as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"ERROR: Unsupported archive format: {archive_path.name}")
            sys.exit(1)

        print(f"Extraction complete.")

    except (zipfile.BadZipFile, tarfile.TarError):
        print(f"ERROR: Corrupted archive file: {archive_path}")
        sys.exit(1)


def download_and_extract_dataset(
    url: str,
    data_dir: Path,
    output_dir: Path,
    archive_filename: str = None,
    cleanup=False,
):
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if extracted data exists
    existing_files = [
        f
        for f in output_dir.iterdir()
        if not (f.name.endswith(".zip") or ".tar" in f.name)
    ]

    if existing_files and not (len(existing_files) == 1 and existing_files[0].is_dir()):
        print(f"✅ Dataset contents already found at '{output_dir}'. Setup complete.")
        return

    # Determine archive filename from URL if not provided
    if not archive_filename:
        archive_filename = url.split("/")[-1]

    archive_path = data_dir / archive_filename

    # Download if archive doesn't exist
    if not archive_path.exists():
        download_file(url, archive_path)
    else:
        print(f"Archive '{archive_path}' exists. Skipping download.")

    # Extract
    extract_archive(archive_path, output_dir)

    # Check directory contents (ignoring the archive file itself)
    extracted_items = [item for item in output_dir.iterdir() if item != archive_path]

    # If there is exactly one item and it is a directory, it's a nested root.
    if len(extracted_items) == 1 and extracted_items[0].is_dir():
        nested_folder = extracted_items[0]
        print(
            f"Detected nested structure: '{nested_folder.name}'. Moving contents up..."
        )

        # Move all items from the nested folder to the output_dir
        for sub_item in nested_folder.iterdir():
            destination = output_dir / sub_item.name
            shutil.move(str(sub_item), str(destination))

        # Remove the now-empty nested folder
        nested_folder.rmdir()
        print(f"Flattening complete. Removed '{nested_folder.name}'.")

    if cleanup:
        print(f"Cleaning up '{archive_path}'...")
        archive_path.unlink()
        print("Cleanup complete.")

    print(f"✅ Dataset ready at '{output_dir}'.\n")
