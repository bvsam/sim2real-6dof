import sys

import canonicalize_models
import download_coco
import download_dtd
import download_modelnet


def main():
    print(f"Starting full environment setup...")

    print("-" * 40)
    print("Processing COCO...")
    try:
        download_coco.main()
    except Exception as e:
        print(f"COCO Failed: {e}")
        sys.exit(1)

    print("-" * 40)
    print("Processing ModelNet...")
    try:
        download_modelnet.main()
    except Exception as e:
        print(f"ModelNet Failed: {e}")
        sys.exit(1)

    print("-" * 40)
    print("Processing DTD...")
    try:
        download_dtd.main()
    except Exception as e:
        print(f"DTD Failed: {e}")
        sys.exit(1)

    print("-" * 40)
    print("Canonicalizing models...")
    try:
        canonicalize_models.main()
    except Exception as e:
        print(f"Model canonicalization Failed: {e}")
        sys.exit(1)

    print("-" * 40)
    print("All setup tasks finished!")


if __name__ == "__main__":
    main()
