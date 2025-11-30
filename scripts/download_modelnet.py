from pathlib import Path

from utils import download_and_extract_dataset

MODELNET_URL = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
BASE_DIR = Path("/home/blender/workspace")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "ModelNet40"


def main():
    print("--- Setting up ModelNet40 Dataset ---")
    download_and_extract_dataset(
        url=MODELNET_URL,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
