# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
"""Converts MS COCO annotations into YOLO labels, next to the images of each split (<split>/images, <split>/labels)"""

import argparse
import json
from pathlib import Path


def convert_coco_to_yolo(coco_json_path, output_dir):
    """Writes one YOLO label file per image, with an empty file for images without annotations"""
    with open(coco_json_path, "r") as file:
        coco_data = json.load(file)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # COCO category IDs are not sequential
    category_mapping = {cat["id"]: i for i, cat in enumerate(coco_data["categories"])}
    image_annotations = {}
    for ann in coco_data["annotations"]:
        image_annotations.setdefault(ann["image_id"], []).append(ann)

    for img in coco_data["images"]:
        label_file = Path(output_dir) / f"{Path(img['file_name']).stem}.txt"
        if img["id"] in image_annotations:
            with open(label_file, "w") as file:
                for ann in image_annotations[img["id"]]:
                    x, y, w, h = ann["bbox"]
                    x_center = (x + w / 2) / img["width"]
                    y_center = (y + h / 2) / img["height"]
                    width = w / img["width"]
                    height = h / img["height"]
                    class_id = category_mapping[ann["category_id"]]
                    file.write(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    )
        else:
            label_file.touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco_root",
        required=True,
        help="MS COCO with annotations/ and train2017/images/, val2017/images/",
    )
    args = parser.parse_args()
    for split in ("train2017", "val2017"):
        convert_coco_to_yolo(
            f"{args.coco_root}/annotations/instances_{split}.json",
            f"{args.coco_root}/{split}/labels",
        )
        print(f"Converted {split}")
