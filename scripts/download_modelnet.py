# download_modelnet.py
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = Path("/home/blender/workspace")
# The directory where the data will be stored, relative to this script.
DATA_DIR = BASE_DIR / "data"
# The final name of the directory after extraction.
MODEL_NAME = "ModelNet40"
# The URL to download the dataset from.
DOWNLOAD_URL = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
# Path to the final extracted directory.
EXTRACTED_DIR_PATH = DATA_DIR / MODEL_NAME
# Path to the temporary zip file.
ZIP_FILE_PATH = DATA_DIR / f"{MODEL_NAME}.zip"


def main():
    """
    Downloads and extracts the ModelNet40 dataset.

    Checks if the dataset already exists. If not, it downloads the zip
    archive, shows a progress bar, extracts it to the 'data/' directory,
    and cleans up the zip file.
    """
    # 1. Check if the dataset already exists to avoid re-downloading.
    if EXTRACTED_DIR_PATH.is_dir():
        print(f"✅ Dataset already found at: '{EXTRACTED_DIR_PATH}'")
        print("Skipping download.")
        return

    print(f"Dataset not found. Starting download and setup...")

    # 2. Create the data directory if it doesn't exist.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created data directory: '{DATA_DIR}'")

    # 3. Download the file with a progress bar.
    try:
        print(f"Downloading from: {DOWNLOAD_URL}")
        response = requests.get(DOWNLOAD_URL, stream=True, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte

        progress_bar = tqdm(
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            desc=f"Downloading {MODEL_NAME}",
        )

        with open(ZIP_FILE_PATH, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("❌ ERROR, something went wrong during download.")
            sys.exit(1)

    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Failed to download file. {e}")
        sys.exit(1)

    # 4. Unzip the file.
    try:
        print(f"\nExtracting '{ZIP_FILE_PATH}'...")
        with zipfile.ZipFile(ZIP_FILE_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print(f"❌ ERROR: Failed to unzip file. It may be corrupted.")
        sys.exit(1)

    # 5. Clean up the downloaded zip file.
    print(f"Cleaning up '{ZIP_FILE_PATH}'...")
    ZIP_FILE_PATH.unlink()  # Use unlink() for Path objects, same as os.remove()

    print("\n✅ Setup complete! The ModelNet40 dataset is ready.")


if __name__ == "__main__":
    main()
