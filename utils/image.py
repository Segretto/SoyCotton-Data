from PIL import Image
import cv2

def parse_resize_arg(resize_arg):
    if resize_arg is None or resize_arg.lower() == "none":
        return None
    try:
        w, h = resize_arg.lower().split("x")
        return (int(w), int(h))
    except ValueError:
        print(f"Invalid format for --resize: '{resize_arg}'. Use 'widthxheight' or 'None'.")
        return None

def load_image(image_path, mode="pil"):
    if mode == "pil":
        return Image.open(image_path)
    elif mode == "cv2":
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image {image_path}")
        return img
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def resize_image(img, resize_dims, mode="pil"):
    if resize_dims is None:
        return img
    w, h = resize_dims
    if mode == "pil":
        return img.resize((w, h), Image.LANCZOS)
    elif mode == "cv2":
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def save_image(img, output_path, mode="pil"):
    if mode == "pil":
        img.save(output_path)
    elif mode == "cv2":
        cv2.imwrite(str(output_path), img)
    else:
        raise ValueError(f"Unsupported mode: {mode}")