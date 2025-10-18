# download_coco.py
# Downloads and extracts the COCO 2017 training dataset for backgrounds.

import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# --- Configuration ---
COCO_URL = "http://images.cocodataset.org/zips/train2017.zip"
BASE_DIR = Path("/home/blender/workspace")
DATA_DIR = BASE_DIR / "data"
ZIP_FILENAME = "train2017.zip"
FINAL_DIR_NAME = "train2017"


def main():
    """
    Main function to download, extract, and clean up the COCO dataset.
    """
    # Create the data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)

    zip_path = DATA_DIR / ZIP_FILENAME
    final_dir_path = DATA_DIR / FINAL_DIR_NAME

    # --- 1. Check if the dataset already exists ---
    # Check if the final unzipped directory exists and is not empty
    if final_dir_path.exists() and any(final_dir_path.iterdir()):
        print(
            f"✅ COCO dataset already found at '{final_dir_path}'. Setup is complete."
        )
        return

    # --- 2. Download the zip file (if it's not already there) ---
    if not zip_path.exists():
        print(f"Downloading COCO dataset from {COCO_URL}...")
        print("This is a large file (18 GB), so it may take a while.")

        try:
            # Use streaming to handle the large file efficiently
            response = requests.get(COCO_URL, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes

            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024 * 1024  # 1 MB chunks

            progress_bar = tqdm(
                total=total_size_in_bytes,
                unit="iB",
                unit_scale=True,
                desc="Downloading train2017.zip",
            )

            with open(zip_path, "wb") as file:
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
            if zip_path.exists():
                zip_path.unlink()  # Clean up partial download
            return
    else:
        print(f"Zip file '{zip_path}' already exists. Skipping download.")

    # --- 3. Unzip the file ---
    print(f"Extracting '{zip_path}'...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print(f"✅ Extraction complete. Dataset is in '{final_dir_path}'.")
    except zipfile.BadZipFile:
        print(
            f"Error: '{zip_path}' is a corrupted zip file. Please delete it and run the script again."
        )
        return

    # --- 4. Clean up the zip file ---
    print(f"Cleaning up by deleting '{zip_path}'...")
    zip_path.unlink()
    print("✅ Cleanup complete.")
    print("\n🎉 COCO dataset setup is finished!")


if __name__ == "__main__":
    main()
