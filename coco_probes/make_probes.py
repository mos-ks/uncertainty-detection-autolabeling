# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
"""Builds the MS COCO composition probes: equally sized image sets drawn from train2017 with weights favoring a composition"""

import argparse
import collections
import os
import random

import numpy as np


def image_statistics(label_dir):
    """Number of objects, their classes, and their mean relative box area per image, from YOLO labels"""
    stats = {}
    for label_file in sorted(os.listdir(label_dir)):
        with open(os.path.join(label_dir, label_file)) as file:
            rows = [line.split() for line in file if line.strip()]
        if not rows:
            continue
        stats[label_file[:-4]] = {
            "n": len(rows),
            "cls": [int(r[0]) for r in rows],
            "mean_area": float(np.mean([float(r[3]) * float(r[4]) for r in rows])),
        }
    return stats


def weighted_sample(keys, weights, n, rng):
    """Samples n distinct keys with probability proportional to their weights (Efraimidis-Spirakis)"""
    weights = np.maximum(np.asarray(weights, float), 1e-9)
    u = np.array([rng.random() for _ in range(len(keys))])
    order = np.argsort(-(u ** (1.0 / weights)))
    return [keys[i] for i in order[:n]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco_train",
        type=str,
        required=True,
        help="train2017 folder in YOLO format, i.e., with images/ and labels/",
    )
    parser.add_argument("--n_images", type=int, default=5900, help="Images per probe")
    parser.add_argument("--seed", type=int, default=20260824, help="Sampling seed")
    args = parser.parse_args()

    stats = image_statistics(os.path.join(args.coco_train, "labels"))
    freq = collections.Counter(c for s in stats.values() for c in s["cls"])
    keys = sorted(stats)
    print(
        f"train2017: {len(keys)} images with objects, {sum(freq.values())} objects, {len(freq)} classes"
    )

    rare = {c for c, _ in sorted(freq.items(), key=lambda kv: kv[1])[:40]}
    head = {c for c, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:10]}
    n_obj = np.array([stats[k]["n"] for k in keys], float)
    area = np.array([stats[k]["mean_area"] for k in keys], float)
    tail_frac = np.array(
        [sum(c in rare for c in stats[k]["cls"]) / max(1, stats[k]["n"]) for k in keys]
    )
    head_frac = np.array(
        [sum(c in head for c in stats[k]["cls"]) / max(1, stats[k]["n"]) for k in keys]
    )
    designs = {
        "random": np.ones(len(keys)),
        "crowded": n_obj**0.45,
        "sparse": n_obj**-0.45,
        "tailbias": (0.15 + tail_frac) ** 1.5,
        "headbias": (0.15 + head_frac) ** 1.5,
        "largeobj": (area + 1e-6) ** 0.18,
        "smallobj": (area + 1e-6) ** -0.18,
    }

    output_dir = os.path.join(args.coco_train, "probes")
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(args.seed)
    for name, weights in designs.items():
        selected = weighted_sample(keys, weights, args.n_images, rng)
        with open(os.path.join(output_dir, f"coco_{name}.txt"), "w") as file:
            for key in selected:
                file.write(f"./images/{key}.jpg\n")
        n_objects = sum(stats[k]["n"] for k in selected)
        n_classes = len({c for k in selected for c in stats[k]["cls"]})
        print(
            f"{name:10s} images={len(selected)} objects={n_objects} classes={n_classes}"
        )


if __name__ == "__main__":
    main()
