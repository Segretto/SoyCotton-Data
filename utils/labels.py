def read_labels(label_file, mode="segmentation", gt=False):
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if mode == "segmentation":
                if len(parts) < 4:
                    print(f"Invalid segmentation label format in {label_file}: {line}")
                    continue
                cls_id = parts[0]
                confidence = 1.0 if gt else float(parts[-1])
                coordinates = parts[1:] if gt else parts[1:-1]
                if len(coordinates) % 2 != 0:
                    print(f"Invalid number of coordinates in {label_file}: {line}")
                    continue
                polygon_coords = [(float(coordinates[2*i]), float(coordinates[2*i+1])) 
                                 for i in range(len(coordinates)//2)]
                labels.append({'class_id': cls_id, 'polygon': polygon_coords, 'confidence': confidence})
            elif mode == "bounding_box":
                if gt and len(parts) == 5:
                    cls, x_center, y_center, width, height = parts
                    labels.append({
                        'class_id': cls, 'x_center': float(x_center), 'y_center': float(y_center),
                        'width': float(width), 'height': float(height)
                    })
                elif not gt and len(parts) == 6:
                    cls, x_center, y_center, width, height, conf = parts
                    labels.append({
                        'class_id': cls, 'x_center': float(x_center), 'y_center': float(y_center),
                        'width': float(width), 'height': float(height), 'confidence': float(conf)
                    })
                else:
                    print(f"Invalid bounding box label format in {label_file}: {line}")
    return labels