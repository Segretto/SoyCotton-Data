from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

def load_fonts(main_size=26, legend_size=36):
    try:
        main_font = ImageFont.truetype("DroidSerif-Regular.ttf", size=main_size)
        legend_font = ImageFont.truetype("DroidSerif-Regular.ttf", size=legend_size)
    except IOError:
        main_font = ImageFont.load_default(size=main_size)
        legend_font = ImageFont.load_default(size=legend_size)
    return main_font, legend_font

def draw_segmentation_masks(img, labels, class_map, gt, main_font, legend_font, alpha=0.6, conf_threshold=0.5):
    img_height, img_width = img.shape[:2]
    overlay = img.copy()
    for label in labels:
        cls_id = label['class_id']
        if cls_id not in class_map:
            continue
        class_info = class_map[cls_id]
        color = class_info['color'][::-1]  # BGR for OpenCV
        confidence = label['confidence']
        if not gt and confidence < conf_threshold:
            continue
        polygon = np.array(label['polygon'], dtype=np.float32) * [img_width, img_height]
        abs_coords = np.round(polygon).astype(np.int32)
        abs_coords = np.clip(abs_coords, [0, 0], [img_width - 1, img_height - 1])
        cv2.fillPoly(overlay, [abs_coords], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGBA')
    draw = ImageDraw.Draw(pil_img)
    if not gt:
        draw_confidence_values(draw, class_map, labels, img_width, img_height, main_font, conf_threshold)
    draw_legend(draw, class_map, legend_font, img_width, img_height)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)

def draw_bounding_boxes(img, labels, class_map, gt, main_font, legend_font):
    img = img.convert('RGBA')
    img_width, img_height = img.size
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for label in labels:
        cls_id = label['class_id']
        if cls_id not in class_map:
            continue
        color = class_map[cls_id]['color']
        x_center = label['x_center'] * img_width
        y_center = label['y_center'] * img_height
        width = label['width'] * img_width
        height = label['height'] * img_height
        x_min, y_min = max(0, x_center - width / 2), max(0, y_center - height / 2)
        x_max, y_max = min(img_width, x_center + width / 2), min(img_height, y_center + height / 2)
        fill_color = color + (40,)  # 15% opacity
        outline_color = color + (255,)
        draw.rounded_rectangle([x_min, y_min, x_max, y_max], radius=10, fill=fill_color, outline=outline_color, width=3)
        if not gt:
            confidence = label['confidence']
            text = f"{confidence:.2f}"
            bbox = main_font.getbbox(text)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            text_pos = (x_min + (x_max - x_min - text_width) // 2, y_min - text_height - 2)
            if text_pos[1] < 0:
                text_pos = (text_pos[0], y_max + 2)
            draw.text(text_pos, text, font=main_font, fill=(255, 255, 255, 255))
    draw_legend(draw, class_map, legend_font, img_width, img_height)
    return Image.alpha_composite(img, overlay).convert('RGB')

def draw_confidence_values(draw, class_map, labels, img_width, img_height, font, conf_threshold):
    # [Your existing draw_confidence_values code, simplified if needed]
    pass

def draw_legend(draw, class_map, font, img_width, img_height, radius=10):
    # [Your existing draw_legend code, unchanged]
    pass