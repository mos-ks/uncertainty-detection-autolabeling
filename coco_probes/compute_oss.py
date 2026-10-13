# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
"""Computes the similarity of the MS COCO composition probes to val2017 as the reference set"""

import argparse
import os
import sys

import cv2
import numpy as np
from scipy.fftpack import dct

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)
from active_learning_eval import Similarity

PROBES = [
    "coco_crowded",
    "coco_headbias",
    "coco_largeobj",
    "coco_random",
    "coco_smallobj",
    "coco_sparse",
    "coco_tailbias",
]


def collect_metrics(split_dir, image_names, classes):
    """Aspect ratio, mean DCT, and mean color histogram of each object crop per class, as in Similarity.collect_metrics"""
    metrics = {cl: [[], [], []] for cl in classes}
    for im_name in image_names:
        label_file = os.path.join(
            split_dir, "labels", os.path.splitext(im_name)[0] + ".txt"
        )
        if not os.path.exists(label_file):
            continue
        image = cv2.cvtColor(
            cv2.imread(os.path.join(split_dir, "images", im_name)), cv2.COLOR_BGR2RGB
        )
        height, width = image.shape[:2]
        with open(label_file) as file:
            rows = [line.split() for line in file if len(line.split()) >= 5]
        for row in rows:
            xc, yc, bw, bh = map(float, row[1:5])
            x1, x2 = max(0, int((xc - bw / 2) * width)), min(
                width, int((xc + bw / 2) * width)
            )
            y1, y2 = max(0, int((yc - bh / 2) * height)), min(
                height, int((yc + bh / 2) * height)
            )
            crop = image[y1:y2, x1:x2, :]
            if crop.size == 0 or np.min([crop.shape[0], crop.shape[1]]) <= 2:
                continue
            cl = int(float(row[0]))
            metrics[cl][0].append(crop.shape[1] / crop.shape[0])
            metrics[cl][1].append(
                np.mean(
                    dct(dct(np.array(crop), axis=0, norm="ortho"), axis=1, norm="ortho")
                )
            )
            metrics[cl][2].append(
                np.mean(
                    cv2.calcHist(
                        crop, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
                    ).flatten()
                )
            )
    return {
        cl: (
            [np.asarray(m[0]), np.asarray(m[1]), np.nan_to_num(np.asarray(m[2]), nan=1)]
            if m[0]
            else []
        )
        for cl, m in metrics.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco_root",
        required=True,
        help="MS COCO with train2017/ (incl. probes/) and val2017/ in YOLO format",
    )
    parser.add_argument(
        "--min_class_crops",
        type=int,
        default=50,
        help="Minimum number of reference crops per class",
    )
    args = parser.parse_args()

    classes = list(range(80))
    train_dir = os.path.join(args.coco_root, "train2017")
    val_dir = os.path.join(args.coco_root, "val2017")
    sets = []
    for probe in PROBES:
        with open(os.path.join(train_dir, "probes", probe + ".txt")) as file:
            image_names = [
                os.path.basename(line.strip()) for line in file if line.strip()
            ]
        sets.append(collect_metrics(train_dir, image_names, classes))
        print(f"Collected crops of {probe}")
    sets.append(
        collect_metrics(
            val_dir, sorted(os.listdir(os.path.join(val_dir, "images"))), classes
        )
    )

    similarity, weighting_active, _ = Similarity.calculate_set_similarity(
        sets, classes, PROBES, min_class_crops=args.min_class_crops
    )
    print(f"Class weighting active: {weighting_active}")
    for probe, value in similarity:
        print(f"{probe}: OSS = {value:.4f}")


if __name__ == "__main__":
    main()
