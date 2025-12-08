import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser(
        description="Upload archived data to Hugging Face Datasets."
    )
    # Directory containing the archives
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to the directory containing .tar or .tar.gz files",
    )
    # Repo ID (e.g., username/dataset_name)
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="The Hugging Face Repo ID (e.g. your-username/my-dataset)",
    )
    # Authentication Token
    parser.add_argument(
        "--token", type=str, required=True, help="Your Hugging Face Write Token"
    )
    # Create Repo Flag
    parser.add_argument(
        "--create",
        action="store_true",
        help="If set, attempts to create the repo (Private) if it does not exist.",
    )
    parser.add_argument(
        "--public",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the repository should be public",
    )
    args = parser.parse_args()

    # Basic Validation
    if not args.input_dir.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    # Initialize API
    api = HfApi(token=args.token)

    # Handle Creation
    if args.create:
        print(f"Ensuring repository '{args.repo_id}' exists...")
        try:
            create_repo(
                repo_id=args.repo_id,
                token=args.token,
                repo_type="dataset",
                private=not args.public,
                exist_ok=True,
            )
            print("Repo ready.")
        except Exception as e:
            print(f"Error creating repo: {e}")
            return

    # Upload
    print(f"Starting upload from '{args.input_dir}' to '{args.repo_id}'...")

    try:
        api.upload_folder(
            folder_path=args.input_dir,
            repo_id=args.repo_id,
            repo_type="dataset",
            path_in_repo="data",  # Stores files in a clean 'data' subfolder
            allow_patterns=[
                "*.tar",
                "*.tar.gz",
                "*.tgz",
            ],
        )
        print("Upload Complete Successfully!")
    except Exception as e:
        print(f"Upload Failed: {e}")


if __name__ == "__main__":
    main()
