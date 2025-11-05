import json
import os
from pathlib import Path
import shutil

# Use the user's home directory for a persistent, central settings file
SETTINGS_PATH = Path.home() / ".snapsearch_settings.json"
print("Settings Path", SETTINGS_PATH)
def get_settings_path() -> Path:
    """Returns the path to the settings file."""
    return SETTINGS_PATH

def load_folders() -> list[str]:
    """Loads the list of managed folder paths from the JSON."""
    if not SETTINGS_PATH.exists():
        return []  # Return empty list if no settings file
    try:
        with open(SETTINGS_PATH, "r") as f:
            data = json.load(f)
            return data.get("managed_folders", [])
    except json.JSONDecodeError:
        print("Error: Settings file is corrupt. Returning empty list.")
        return []

def save_folders(folder_list: list[str]):
    """Saves the complete list of managed folders to the JSON."""
    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump({"managed_folders": folder_list}, f, indent=2)
    except Exception as e:
        print(f"Error saving settings: {e}")

def add_folder(folder_path: str):
    """Adds a new folder path to the managed list."""
    folders = load_folders()
    if folder_path not in folders:
        folders.append(folder_path)
        save_folders(folders)
        print(f"Added new folder to manager: {folder_path}")

def remove_folder(folder_path: str):
    """Removes a folder path from the managed list."""
    folders = load_folders()
    if folder_path in folders:
        folders.remove(folder_path)
        save_folders(folders)
        print(f"Removed folder from manager: {folder_path}")


def delete_folder_permanently(folder_path: str):
    """
    Permanently deletes a folder from the file system and
    removes it from the managed list.
    """
    if not folder_path or not os.path.exists(folder_path):
        print(f"Error: Folder path not valid: {folder_path}")
        return

    # 1. Permanently delete the folder and all its contents
    print(f"Deleting folder from disk: {folder_path}")
    try:
        shutil.rmtree(folder_path)
    except Exception as e:
        print(f"Error deleting folder contents: {e}")
        # We'll still try to remove it from the manager

    # 2. Remove the folder from the settings.json
    remove_folder(folder_path)
    print(f"Folder deletion complete.")