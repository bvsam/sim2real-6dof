from pathlib import Path

from utils import download_and_extract_dataset

COCO_URL = "http://images.cocodataset.org/zips/train2017.zip"
BASE_DIR = Path("/home/blender/workspace")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "train2017"


def main():
    print("--- Setting up COCO 2017 Dataset ---")
    download_and_extract_dataset(
        url=COCO_URL,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
