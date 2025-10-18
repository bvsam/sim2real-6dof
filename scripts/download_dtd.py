# download_dtd.py
# Downloads and extracts the DTD (Describable Textures Dataset) for textures.

import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

# --- Configuration ---
DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
BASE_DIR = Path("/home/blender/workspace")
DATA_DIR = BASE_DIR / "data"
TAR_FILENAME = "dtd-r1.0.1.tar.gz"
FINAL_DIR_NAME = "dtd"


def main():
    """
    Main function to download, extract, and clean up the DTD dataset.
    """
    # Create the data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)

    tar_path = DATA_DIR / TAR_FILENAME
    final_dir_path = DATA_DIR / FINAL_DIR_NAME

    # --- 1. Check if the dataset already exists ---
    # Check if the final unzipped directory exists and is not empty
    if final_dir_path.exists() and any(final_dir_path.iterdir()):
        print(f"✅ DTD dataset already found at '{final_dir_path}'. Setup is complete.")
        return

    # --- 2. Download the tar.gz file (if it's not already there) ---
    if not tar_path.exists():
        print(f"Downloading DTD dataset from {DTD_URL}...")
        print("This file is about 600 MB, so it may take a few minutes.")

        try:
            # Use streaming to handle the file efficiently
            response = requests.get(DTD_URL, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes

            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024 * 1024  # 1 MB chunks

            progress_bar = tqdm(
                total=total_size_in_bytes,
                unit="iB",
                unit_scale=True,
                desc="Downloading dtd-r1.0.1.tar.gz",
            )

            with open(tar_path, "wb") as file:
                for data_chunk in response.iter_content(block_size):
                    progress_bar.update(len(data_chunk))
                    file.write(data_chunk)
            progress_bar.close()

            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong during download.")
                return

            print("✅ Download complete.")

        except requests.exceptions.RequestException as e:
            print(f"Error during download: {e}")
            if tar_path.exists():
                tar_path.unlink()  # Clean up partial download
            return
    else:
        print(f"Tar file '{tar_path}' already exists. Skipping download.")

    # --- 3. Extract the tar.gz file ---
    print(f"Extracting '{tar_path}'...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar_ref:
            tar_ref.extractall(DATA_DIR)
        print(f"✅ Extraction complete. Dataset is in '{final_dir_path}'.")
    except tarfile.TarError:
        print(
            f"Error: '{tar_path}' is a corrupted tar file. Please delete it and run the script again."
        )
        return

    # --- 4. Clean up the tar.gz file ---
    print(f"Cleaning up by deleting '{tar_path}'...")
    tar_path.unlink()
    print("✅ Cleanup complete.")
    print("\n🎉 DTD dataset setup is finished!")


if __name__ == "__main__":
    main()
