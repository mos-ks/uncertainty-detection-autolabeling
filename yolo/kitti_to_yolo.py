# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
"""Converts KITTI object labels into YOLO format and writes the validation image list"""

import argparse
import os

from PIL import Image

CLASSES = ["Pedestrian", "Car", "Van", "Cyclist", "Truck", "Tram", "Person_sitting"]


def kitti_to_yolo_lines(label_file, width, height):
    """YOLO lines (class, normalized center and size) of the KITTI objects in one label file"""
    dw, dh = 1.0 / width, 1.0 / height
    lines = []
    with open(label_file) as file:
        for row in file:
            fields = row.split()
            if not fields or fields[0] not in CLASSES:
                continue
            x1, y1, x2, y2 = map(float, fields[4:8])
            lines.append(
                f"{CLASSES.index(fields[0])} {(x1 + x2) / 2.0 * dw} {(y1 + y2) / 2.0 * dh} "
                f"{(x2 - x1) * dw} {(y2 - y1) * dh}"
            )
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kitti_training",
        type=str,
        required=True,
        help="KITTI training folder with image_2/ and label_2/",
    )
    parser.add_argument(
        "--val_index_list",
        type=str,
        required=True,
        help="datasets/KITTI/vaL_index_list.txt",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="YOLO dataset root to create, with images/ and labels/",
    )
    args = parser.parse_args()

    image_dir = os.path.join(args.kitti_training, "image_2")
    label_dir = os.path.join(args.output, "labels")
    os.makedirs(label_dir, exist_ok=True)
    if not os.path.exists(os.path.join(args.output, "images")):
        os.symlink(os.path.abspath(image_dir), os.path.join(args.output, "images"))

    im_names = sorted(os.listdir(image_dir))
    for im_name in im_names:
        width, height = Image.open(os.path.join(image_dir, im_name)).size
        stem = os.path.splitext(im_name)[0]
        lines = kitti_to_yolo_lines(
            os.path.join(args.kitti_training, "label_2", stem + ".txt"), width, height
        )
        with open(os.path.join(label_dir, stem + ".txt"), "w") as file:
            file.write("".join(line + "\n" for line in lines))

    with open(args.val_index_list) as file:
        val_indices = [int(line.strip()) for line in file if line.strip()]
    with open(os.path.join(args.output, "validation.txt"), "w") as file:
        for index in val_indices:
            file.write(f"./images/{im_names[index]}\n")
    print(
        f"Converted {len(im_names)} label files, {len(val_indices)} validation images -> {args.output}"
    )


if __name__ == "__main__":
    main()
