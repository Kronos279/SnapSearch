
import os
import uuid
import re


folder_path = "/Volumes/Micron/Images"
extension = (".png", ".jpg", ".jpeg", ".webp", ".heic")
uuid_regex = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\.\w+$")

import os
import uuid
import re
from PIL import Image
from pillow_heif import register_heif_opener

# Enable HEIC/HEIF support in Pillow
register_heif_opener()

# folder_path = "/Volumes/Micron/Images"
extensions = (".png", ".jpg", ".jpeg", ".webp", ".heic")

# Regex to detect if a filename already follows UUID pattern
uuid_regex = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\.\w+$"
)

def renameAllImages(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Skip folders
        if not os.path.isfile(file_path):
            continue

        # Skip if already renamed (UUID)
        if uuid_regex.match(filename):
             continue

        # Process only image files with valid extensions
        if filename.lower().endswith(extensions):
            unique_id = str(uuid.uuid4())
            ext = os.path.splitext(filename)[1].lower()
            new_filename = f"{unique_id}{ext}"
            new_path = os.path.join(folder_path, new_filename)

            # Rename first
            os.rename(file_path, new_path)
            print(f"Renamed: {filename} → {new_filename}")

            # Convert HEIC/WEBP/PNG → JPG
            if ext in [".heic", ".webp", ".png"]:
                try:
                    img = Image.open(new_path).convert("RGB")
                    jpg_path = os.path.join(folder_path, f"{unique_id}.jpg")
                    img.save(jpg_path, "JPEG", quality=95)

                    os.remove(new_path)
                    print(f" Converted: {new_filename} → {unique_id}.jpg")

                except Exception as e:
                    print(f" Failed to convert {new_filename}: {e}")