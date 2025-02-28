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
    colors = []
    for _, cls_info in class_map.items():
        colors.append(cls_info['color'][::-1] + (255,))
    
    for label in labels:
        cls_id = label['class_id']
        if cls_id not in class_map:
            continue

        confidence = label['confidence']

        if confidence < conf_threshold:
            continue

        abs_polygon = np.array(
            [[int(x * img_width), int(y * img_height)] for x, y in label['polygon']],
            dtype=np.int32
        )

        # Calculate bounding box for the polygon
        x_min = np.min(abs_polygon[:, 0])
        y_min = np.min(abs_polygon[:, 1])
        x_max = np.max(abs_polygon[:, 0])
        y_max = np.max(abs_polygon[:, 1])

        # Prepare text
        text = f"{confidence:.2f}"

        # Use font.getbbox() to get the size of the text
        x_left, y_top, x_right, y_bottom = font.getbbox(text)
        text_width = abs(x_right - x_left)
        text_height = abs(y_bottom - y_top)

        bbox_xmid = (x_max - x_min)/2
        bbox_ymid = (y_max - y_min)/2

        # text_position = (x_min + bbox_xmid - text_width/2, y_min - text_height*1.161)
        text_position = (x_min + bbox_xmid - text_width//2, y_min + bbox_ymid - text_height//2)
        text_position_back = (x_min, y_min - text_height - 6)

        # Ensure text is within image bounds
        if text_position[1] < 0:
            text_position = (x_min, y_max + 4)

        # Define shadow offset and color
        shadow_offset = (1, 1)  # (x_offset, y_offset)
        shadow_color = (0, 0, 0, 128)  # Semi-transparent black

        # Draw shadow text on the overlay
        shadow_position = (text_position[0] + shadow_offset[0], text_position[1] + shadow_offset[1])
        draw.text(
            shadow_position,
            text,
            font=font,
            fill=colors[1]
        )

        # Draw text on the overlay
        draw.text(
            text_position,
            text,
            font=font,
            fill=colors[0]  # Bright white color with full opacity
        )

def draw_legend(draw, class_map, font, img_width, img_height, radius=10):
    legend_x = 10  # Padding from the left edge
    legend_y = 10  # Padding from the top edge

    y_text_offset = 5
    x_text_offset = 5

    max_text_width = 0
    total_text_height = 0
    entries = []

    for cls_id, class_info in class_map.items():
        class_name = class_info['name'].capitalize()
        color = class_info['color']
        text = class_name

        # Get text size
        x_left, y_top, x_right, y_bottom = font.getbbox(text)
        text_width = abs(x_right - x_left)
        text_height = abs(y_bottom - y_top)

        max_text_width = max(max_text_width, text_width)
        total_text_height += text_height + 5  # Adding spacing between entries

        entries.append({
            'text': text,
            'text_width': text_width,
            'text_height': text_height,
            'color': color
        })

    # Square size is 80% of text heightimages_folder/
    square_size = int(0.6 * entries[-1]["text_height"])

    # Background for the legend (optional)
    legend_width =  square_size + max_text_width + 5*x_text_offset  
    legend_height = total_text_height + 3*y_text_offset 
    legend_background = [
        (legend_x, legend_y),
        (legend_x + legend_width, legend_y + legend_height)
    ]
    draw.rounded_rectangle(legend_background, radius=radius, fill=(50, 50, 50, 180))

    # Draw each legend entry
    current_y = legend_y + y_text_offset  # Starting y position with padding
    for entry in entries:
        text = entry['text']
        text_width = entry['text_width']
        text_height = entry['text_height']
        color = entry['color']

        square_offset = abs(text_height + 2*y_text_offset - square_size)/2
        square_y = current_y + square_offset

        # Draw the color square
        square_coords = [
            legend_x + x_text_offset,
            square_y,
            legend_x + x_text_offset + square_size,
            square_y + square_size
        ]
        draw.rounded_rectangle(square_coords, radius=radius*0.1, fill=color + (255,), outline=None)

        text_position = (legend_x + x_text_offset*3 + square_size, current_y)
        draw.text(
            text_position,
            text,
            fill=(255, 255, 255, 255), 
            font=font
        )

        current_y += text_height + y_text_offset
