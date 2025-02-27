from pathlib import Path

def validate_paths(images_folder, labels_folder, output_folder):
    images_folder = Path(images_folder)
    labels_folder = Path(labels_folder)
    output_folder = Path(output_folder)
    if not images_folder.exists():
        print(f"Images folder {images_folder} does not exist.")
        return None, None, None
    if not labels_folder.exists():
        print(f"Labels folder {labels_folder} does not exist.")
        return None, None, None
    output_folder.mkdir(parents=True, exist_ok=True)
    return images_folder, labels_folder, output_folder

def get_image_files(images_folder):
    image_extensions = ['.jpg', '.jpeg', '.png']
    images = [f for f in images_folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    if not images:
        print(f"No images found in {images_folder}.")
    return images if images else None