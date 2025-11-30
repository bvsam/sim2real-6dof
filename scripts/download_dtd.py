from pathlib import Path

from utils import download_and_extract_dataset

DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
BASE_DIR = Path("/home/blender/workspace")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "dtd"


def main():
    print("--- Setting up DTD Dataset ---")
    download_and_extract_dataset(
        url=DTD_URL,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
