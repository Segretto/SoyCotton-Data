# Leaf-Level Soybean and Cotton Dataset Repository

This repository is dedicated to managing and processing the *SoyCotton* dataset, a comprehensive collection of leaf-level annotations for soybean and cotton plants. The dataset includes 640 high-resolution images with over 12,000 annotated leaves (7,221 soybean and 5,190 cotton), captured across diverse growth stages, weed pressures, and lighting conditions. It supports both bounding-box detection and instance segmentation tasks, making it ideal for developing AI-driven solutions for crop management, such as targeted weed control and pest monitoring in soybean-cotton rotation systems.

<div align="center">
  <img src="soy-cotton.png" alt="SoyCotton Dataset Example" width="70%">
</div>

- **Dataset Access**: The full dataset is publicly available on Figshare: [https://figshare.com/articles/preprint/SoyCotton-Leafs/28466636](https://figshare.com/articles/preprint/SoyCotton-Leafs/28466636).
- **Paper Citation**: For detailed methodology and validation results, refer to our paper on arXiv: "A Leaf-Level Dataset for Soybean-Cotton Detection and Segmentation" [http://arxiv.org/abs/2503.01605](http://arxiv.org/abs/2503.01605).

# Tools for Handling the SoyCotton Dataset

This repository provides three Python scripts to help you:

1. **Convert** COCO-style annotations to YOLO-style bounding boxes or segmentation masks (`coco2yolo.py`).  
2. **Split** a COCO dataset into training, validation, and testing sets or create k-fold cross-validation folds (`split_data.py`).  
3. **Visualize** the resulting YOLO annotations on images, either as **bounding boxes** or **segmentation masks** (`render.py`).  

These scripts assume a directory structure typical of COCO datasets and will produce outputs that integrate well with common YOLO-training frameworks.

---

## Table of Contents

- [Leaf-Level Soybean and Cotton Dataset Repository](#leaf-level-soybean-and-cotton-dataset-repository)
- [Tools for Handling the SoyCotton Dataset](#tools-for-handling-the-soycotton-dataset)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Scripts Overview](#scripts-overview)
    - [1. Split Data](#1-split-data)
    - [2. COCO to YOLO format](#2-coco-to-yolo-format)
    - [3. Rendering Boxes and Masks](#3-rendering-boxes-and-masks)
      - [What It Does](#what-it-does)
  - [Training YOLO](#training-yolo)
    - [Detection](#detection)
    - [Segmentation](#segmentation)
  - [Citation](#citation)

---

## Requirements

Libraries for dataset manipulation:

```bash
pip install numpy opencv-python pycocotools joblib pandas scikit-learn
```

For training a yolo11 model, use ultralytics package:

```bash
pip install ultralytics
```

---

## Scripts Overview

### 1. Split Data

Use the script `split_data.py` to divide your COCO dataset into different data splits (train/val/test) or perform k-fold cross-validation, as well as create ablation subsets.

**Typical Tasks**

1. **Standard Train-Val-Test Split**  
   - Control the proportions of your dataset by specifying `--train_ratio` (e.g., 0.7) and `--val_ratio` (e.g., 0.2).  
   - The remainder is automatically assigned to the test set.

**Example**:

```bash
python scripts/split_data.py /path/to/all_images /path/to/coco.json /path/to/output_dir --train_ratio 0.7 --val_ratio 0.2
```
This will create:

```
output_dir/
    ├── images/
        ├── train/
        ├── val/
        └── test/
    └── labels/
        ├── train/coco.json
        ├── val/coco.json
        └── test/coco.json
```

2. **K-Fold Cross-Validation**  
- Use the `--k` argument (e.g., `--k 5`) to build multiple folds.  
- Each fold has its own train/val splits, stored in separate subdirectories.

**Example**:

```bash
python scripts/split_data.py /path/to/all_images /path/to/coco.json /path/to/output_dir --k 5
```

You’ll see five fold-specific directories, each with its own images and labels.

3. **Data Ablation Studies**  
- Specify `--ablation N` to create `N` progressively larger data subsets for experimentation.  
- Each ablation chunk has its own directory, mirroring train/val splits but with fewer samples.

**Example**:

```bash
python scripts/split_data.py /path/to/all_images /path/to/coco.json /path/to/output_dir --ablation 3
```
Generates multiple splits labeled by incremental sizes (e.g., 33%, 66%, etc.).

**Other Key Arguments**:
- `--classes`: Filter the dataset to only include specified class names.  
- `--rename_images`: Rename files for each subset (avoids collisions; defaults to True).  

---

### 2. COCO to YOLO format

Once your dataset is split or organized, run `coco2yolo.py` to convert COCO-style bounding boxes or segmentation masks into YOLO `.txt` files.

**Key Arguments**:
- `dataset_path`: The root directory containing `images/` and `labels/<split>/coco.json`.  
- `--mode`:  
- `detection` for bounding boxes  
- `segmentation` for segmentation polygons  
- `--custom_data_path`: (Optional) Override the path written to the generated `.yaml` file.

**Basic Usage**:

```bash
python scripts/coco2yolo.py /path/to/output_dir --mode detection
```

- Looks for `coco.json` inside each `labels/<split>/` directory.
- Creates YOLO-format `.txt` files (one per image) inside the same split directory.
- Optionally generates a `.yaml` file referencing your train/val directories and listing class names.

That’s all you need to turn your COCO dataset into YOLO labels for common object detection or segmentation tasks!


### 3. Rendering Boxes and Masks

`render.py` renders YOLO-style bounding boxes or segmentation masks on images for quick visual debugging.

**Key arguments**:
- `images_folder`: Path to the folder with input images.  
- `labels_folder`: Path to the folder with `.txt` YOLO label files.  
- `output_folder`: Directory to save rendered images.  
- `--mode`: Choose between `segmentation` or `bounding_box`.  
- `--resize`: Resize the image for visualization (e.g. `--resize 640x480` or `None`).

**Basic usage**:

```bash
python scripts/render.py /path/to/images /path/to/labels /path/to/output --mode segmentation
```

#### What It Does

1. Loads each image in `images_folder`.  
2. Reads the corresponding `.txt` file from `labels_folder` (must match image file name).  
3. Draws bounding boxes (mode=`bounding_box`) or segmentation polygons (mode=`segmentation`).  
4. Saves the rendered images in `output_folder`.


## Training YOLO

To replicate our performance values, we suggest running the yolo training scripts with the optimized hyperparameters. Make sure to add the correct path to the dataset into the shell scripts:

### Detection

```bash
bash scripts/yolo_train_od.sh
```

### Segmentation
```bash
bash scripts/yolo_train_seg.sh
```
## Citation

To cite the dataset or the associated research, please use the following BibTeX entry:

```bibtex
@article{segreto2025leaf,
  author = {Segreto, Thiago H. and Negri, Juliano and Polegato, Paulo H. and Pinheiro, João Manoel Herrera and Godoy, Ricardo and Becker, Marcelo},
  title = {A Leaf-Level Dataset for Soybean-Cotton Detection and Segmentation},
  journal = {arXiv preprint},
  year = {2025},
  url = {http://arxiv.org/abs/2503.01605} 
}