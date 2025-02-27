import argparse
from utils.io import validate_paths, get_image_files
from utils.image import parse_resize_arg, load_image, resize_image, save_image
from utils.labels import read_labels
from utils.visualize import load_fonts, draw_segmentation_masks, draw_bounding_boxes

def main(images_folder, labels_folder, output_folder, resize, mode, gt=False):
    images_folder, labels_folder, output_folder = validate_paths(images_folder, labels_folder, output_folder)
    if not images_folder:
        return
    resize_dims = parse_resize_arg(resize)
    class_map = {'0': {'name': 'soy', 'color': (252, 236, 3)}, '1': {'name': 'cotton', 'color': (201, 14, 230)}}
    process_images(images_folder, labels_folder, output_folder, resize_dims, mode, gt, class_map)

def process_images(images_folder, labels_folder, output_folder, resize_dims, mode, gt, class_map):
    images = get_image_files(images_folder)
    if not images:
        return
    main_font, legend_font = load_fonts(main_size=32 if mode == "bounding_box" else 26)
    img_mode = "cv2" if mode == "segmentation" else "pil"
    for image_path in images:
        label_file = labels_folder / (image_path.stem + '.txt')
        if not label_file.exists():
            print(f"Label file {label_file} does not exist. Skipping.")
            continue
        labels = read_labels(label_file, mode=mode, gt=gt)
        img = load_image(image_path, mode=img_mode)
        img = resize_image(img, resize_dims, mode=img_mode)
        if mode == "segmentation":
            img_with_vis = draw_segmentation_masks(img, labels, class_map, gt, main_font, legend_font)
        elif mode == "bounding_box":
            img_with_vis = draw_bounding_boxes(img, labels, class_map, gt, main_font, legend_font)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        output_path = output_folder / image_path.name
        save_image(img_with_vis, output_path, mode=img_mode)
        print(f"Saved image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render visualizations on images.")
    parser.add_argument("images_folder", help="Path to images folder.")
    parser.add_argument("labels_folder", help="Path to labels folder.")
    parser.add_argument("output_folder", help="Path to output folder.")
    parser.add_argument("--mode", choices=["segmentation", "bounding_box"], default="segmentation",
                        help="Visualization mode: 'segmentation' or 'bounding_box'.")
    parser.add_argument("--gt", action='store_true', help="Use ground truth labels (no confidence scores).")
    parser.add_argument("--resize", default="None", help="Resize images to 'widthxheight' or 'None'.")
    args = parser.parse_args()
    main(**vars(args))