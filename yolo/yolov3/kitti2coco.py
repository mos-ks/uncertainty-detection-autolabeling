#!/usr/bin/env python3

"""
Script to convert KITTI labels (in label_2 folder) to COCO format JSON for evaluation.
The script:
1) Reads a folder containing:
   - image_2/ (with image files)
   - label_2/ (with .txt annotation files, same filename as corresponding images)
2) Reads a val_index_list.txt, which contains indices (one per line) for the validation set,
   where images in image_2 are sorted by name and indexed from 0..N-1.
3) Generates a COCO-format JSON file containing only the annotations for those validation images.

Usage example (from the repository root):
    python yolo/yolov3/kitti2coco.py --data_dir datasets/KITTI/training \
                                     --val_index_list datasets/KITTI/vaL_index_list.txt \
                                     --output_json datasets/KITTI/output_coco.json
"""

import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert KITTI format annotations to COCO format (for evaluation)."
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory containing image_2/ and label_2/.')
    parser.add_argument('--val_index_list', type=str, required=True,
                        help='Path to the text file containing validation indices (one index per line).')
    parser.add_argument('--output_json', type=str, default='coco_annotations.json',
                        help='Output path for the COCO-format JSON file.')
    return parser.parse_args()

def load_val_indices(val_index_list_path):
    """
    Loads validation indices from a text file. One index per line.
    """
    val_indices = []
    with open(val_index_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                val_indices.append(int(line))
    return set(val_indices)

def get_image_file_list(img_dir):
    """
    Returns a sorted list of image filenames in the image_2 folder.
    """
    valid_exts = {'.png', '.jpg', '.jpeg'}
    files = [
        f for f in os.listdir(img_dir) 
        if os.path.splitext(f.lower())[-1] in valid_exts
    ]
    files.sort()  # sort by filename
    return files

def kitti_label_to_bbox(label_line):
    """
    Parse a single line of KITTI label.
    Typical format (for 2D boxes) is:
      type truncation occlusion alpha x1 y1 x2 y2 ... 
    We'll return a dictionary with:
      - class_name
      - [x1, y1, width, height]
    We ignore the rest of the 3D information for COCO's 2D bounding boxes.
    """
    parts = label_line.split()
    
    # The KITTI format (for the first 8 fields):
    # 0  -> type (e.g. Car, Pedestrian, Cyclist)
    # 1  -> truncation
    # 2  -> occlusion
    # 3  -> alpha
    # 4  -> bbox left (x1)
    # 5  -> bbox top (y1)
    # 6  -> bbox right (x2)
    # 7  -> bbox bottom (y2)
    class_name = parts[0]
    x1 = float(parts[4])
    y1 = float(parts[5])
    x2 = float(parts[6])
    y2 = float(parts[7])
    
    # Convert to COCO [x, y, width, height]
    width = x2 - x1
    height = y2 - y1

    return {
        'class_name': class_name,
        'bbox': [x1, y1, width, height]
    }

def main():
    args = parse_args()

    data_dir = args.data_dir
    img_dir = os.path.join(data_dir, 'image_2')
    label_dir = os.path.join(data_dir, 'label_2')
    
    # Load validation indices
    val_indices = load_val_indices(args.val_index_list)

    # Get all image files (sorted by filename)
    image_files = get_image_file_list(img_dir)

    # Categories you want to include, map from class_name to category ID.
    # Expand or modify as you see fit.
    # You can also build this dynamically if you want, but for COCO
    # evaluation, it's typical to have a fixed set of categories.
    category_mapping = {
        'Car': 1,
        'Van': 2,
        'Truck': 3,
        'Pedestrian': 4,
        'Person_sitting': 5,
        'Cyclist': 6,
        'Tram': 7
    }
    
    # Prepare COCO-style container
    coco_output = {
        'info': {
            'description': "KITTI to COCO conversion",
            'url': "",
            'version': "1.0",
            'year': 2023,
            'contributor': "",
            'date_created': ""
        },
        'licenses': [
            {
                'id': 1,
                'name': 'Unknown',
                'url': 'Unknown'
            }
        ],
        'images': [],
        'annotations': [],
        'categories': []
    }

    # Add category information
    for cat_name, cat_id in category_mapping.items():
        coco_output['categories'].append({
            'id': cat_id,
            'name': cat_name,
            'supercategory': 'object'
        })

    annotation_id = 1  # incremental ID for each bounding box
    
    # Loop over all images; filter by indices in val_indices
    for idx, img_filename in enumerate(image_files):
        if idx not in val_indices:
            continue  # skip if not in validation set

        # Build image info
        image_id = idx  # or any unique ID. We'll just use the index.
        image_path = os.path.join(img_dir, img_filename)

        # You may want to use an external library like OpenCV or PIL to get the actual
        # width/height of the image. For demonstration, let's do it with PIL.
        from PIL import Image
        with Image.open(image_path) as im:
            width, height = im.size

        coco_output['images'].append({
            'file_name': img_filename,
            'height': height,
            'width': width,
            'id': image_id
        })

        # Corresponding label file
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)

        # If label file doesn’t exist (e.g., some images have no labels), skip
        if not os.path.exists(label_path):
            continue

        with open(label_path, 'r') as lf:
            lines = lf.readlines()

        # Parse each bounding box line
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parsed = kitti_label_to_bbox(line)
            class_name = parsed['class_name']

            # Skip if the class_name is not in your mapping
            if class_name not in category_mapping:
                continue

            cat_id = category_mapping[class_name]
            [x, y, w, h] = parsed['bbox']

            # Skip invalid boxes (width or height <= 0)
            if w <= 0 or h <= 0:
                continue

            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': cat_id,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0
            }
            coco_output['annotations'].append(annotation)
            annotation_id += 1

    # Save to JSON
    with open(args.output_json, 'w') as out_file:
        json.dump(coco_output, out_file, indent=2)

    print(f"COCO-style annotations saved to {args.output_json}")


if __name__ == '__main__':
    main()
