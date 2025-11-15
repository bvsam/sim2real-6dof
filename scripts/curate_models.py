# curate_models.py
import os
import pathlib
import sys
from pathlib import Path

import trimesh

# --- Configuration ---
# IMPORTANT: Change this to the path where your ModelNet40 dataset is located.
# It should be the parent directory containing the "cup" folder.
DEFAULT_BASE_DIR = Path("/home/blender/workspace")
MODELNET_PATH = DEFAULT_BASE_DIR / "data" / "ModelNet40"
CATEGORY = "cup"
OUTPUT_FILE = DEFAULT_BASE_DIR / "annotations" / "approved_mugs_with_handles.txt"
# ---------------------


def clear_screen():
    """Clears the console screen."""
    os.system("cls" if os.name == "nt" else "clear")


def find_model_files(base_path, category):
    """Finds all .off files in the train and test directories for a category."""
    category_path = pathlib.Path(base_path) / category
    if not category_path.exists():
        print(f"Error: Directory not found at '{category_path}'")
        print("Please make sure MODELNET_PATH is set correctly.")
        sys.exit(1)

    print(f"Searching for models in '{category_path}'...")
    train_path = category_path / "train"
    test_path = category_path / "test"

    files = list(train_path.glob("*.off")) + list(test_path.glob("*.off"))

    if not files:
        print(f"Error: No '.off' files found in '{train_path}' or '{test_path}'.")
        print("Please check your dataset structure.")
        sys.exit(1)

    return sorted(files)


def main():
    """Main function to run the curation process."""
    all_files = find_model_files(MODELNET_PATH, CATEGORY)
    total_files = len(all_files)
    print(f"Found {total_files} models to review.")

    selected_files = set()
    current_index = 0

    while current_index < total_files:
        filepath = all_files[current_index]

        # --- Display Model ---
        try:
            # force='mesh' ensures we get a mesh object, ignoring textures etc.
            mesh = trimesh.load(str(filepath), force="mesh")

            # Create a scene and add the mesh. Trimesh automatically centers it.
            scene = trimesh.Scene(mesh)

            # This line is blocking. The script will pause until you close the window.
            scene.show(window_title=f"Model: {filepath.name}")

        except Exception as e:
            print(
                f"\n[!] Warning: Could not load or display {filepath.name}. Skipping. Error: {e}"
            )
            current_index += 1
            continue

        # --- Get User Input ---
        while True:
            clear_screen()
            print("--- Model Curation Tool ---")
            print(f"Viewing model {current_index + 1} of {total_files}")
            print(f"File: {filepath.name}")
            print(f"\nTotal selected: {len(selected_files)}")

            status = "SELECTED" if str(filepath) in selected_files else "NOT SELECTED"
            print(f"Current model status: {status}")

            print("\nClose the 3D viewer window to enter commands here.")
            print("Controls:")
            print("  (s) - Select this model (approve it)")
            print("  (n) - Next model (skip)")
            print("  (p) - Previous model")
            print("  (q) - Quit and Save")

            choice = input("Enter command: ").lower().strip()

            if choice == "s":
                selected_files.add(str(filepath))
                print(f"'{filepath.name}' added to selection.")
                current_index += 1
                break
            elif choice == "n":
                current_index += 1
                break
            elif choice == "p":
                if current_index > 0:
                    current_index -= 1
                else:
                    print("Already at the first model.")
                break
            elif choice == "q":
                # Exit the main loop
                current_index = total_files + 1
                break
            else:
                print("Invalid command. Please try again.")

    # --- Save Results ---
    if selected_files:
        print(
            f"\nCuration complete. Saving {len(selected_files)} selected model paths to '{OUTPUT_FILE}'..."
        )
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            for path in sorted(list(selected_files)):
                f.write(path + "\n")
        print("Done!")
    else:
        print("\nCuration complete. No models were selected.")


if __name__ == "__main__":
    main()
