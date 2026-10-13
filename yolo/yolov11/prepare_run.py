# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
"""Prepares a YOLOv11 run on a KITTI AL selection or an MS COCO composition probe: dataset root, dataset config and trainer config"""

import argparse
import json
import os
import shutil

import yaml

KITTI_NAMES = ["Pedestrian", "Car", "Van", "Cyclist", "Truck", "Tram", "Person_sitting"]
# Fine-tuning on the KITTI AL selections
KITTI_TRAINING = dict(
    model="yolo11s.pt",
    pretrained=True,
    epochs=100,
    batch=32,
    imgsz=640,
    seed=0,
    deterministic=True,
    val=True,
    plots=False,
)
# Training from scratch on the MS COCO composition probes
COCO_TRAINING = dict(
    model="yolo11s.yaml",
    pretrained=False,
    epochs=150,
    batch=32,
    imgsz=640,
    seed=0,
    deterministic=True,
    val=True,
    plots=False,
)


def link(target, link_path):
    """Symlinks a folder, so that each run has its own dataset root and label cache"""
    if not os.path.islink(link_path) and not os.path.exists(link_path):
        os.symlink(os.path.abspath(target), link_path)


def write_configs(run_root, name, names, train_list, val_list, training, runs_dir):
    data = dict(
        names=dict(enumerate(names)),
        path=os.path.abspath(run_root),
        train=train_list,
        val=val_list,
    )
    with open(os.path.join(run_root, "data.yaml"), "w") as file:
        yaml.safe_dump(data, file, sort_keys=False)
    config = dict(
        training,
        data=os.path.abspath(os.path.join(run_root, "data.yaml")),
        project=os.path.abspath(runs_dir),
        name=name,
    )
    config_path = os.path.join(run_root, "train.yaml")
    with open(config_path, "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    print(f"Prepared {name}: python yolo/yolov11/train.py --config {config_path}")


def prepare_kitti(args):
    """KITTI AL selection: AL index file of active_learning_loop.py -> list of the selected images"""
    run_root = os.path.join(args.output_dir, args.name)
    os.makedirs(run_root, exist_ok=True)
    for folder in ("images", "labels"):
        link(os.path.join(args.yolo_root, folder), os.path.join(run_root, folder))
    im_names = sorted(os.listdir(os.path.join(args.yolo_root, "images")))
    with open(args.index_file) as file:
        indices = [int(line.strip()) for line in file if line.strip()]
    with open(os.path.join(run_root, "train.txt"), "w") as file:
        for index in indices:
            file.write(f"./images/{im_names[index]}\n")
    shutil.copyfile(
        os.path.join(args.yolo_root, "validation.txt"),
        os.path.join(run_root, "validation.txt"),
    )
    write_configs(
        run_root,
        args.name,
        KITTI_NAMES,
        "train.txt",
        "validation.txt",
        KITTI_TRAINING,
        args.runs_dir,
    )


def prepare_coco(args):
    """MS COCO composition probe of make_probes.py, evaluated on val2017"""
    run_root = os.path.join(args.output_dir, args.probe)
    os.makedirs(os.path.join(run_root, "val"), exist_ok=True)
    for folder in ("images", "labels"):
        link(
            os.path.join(args.coco_root, "train2017", folder),
            os.path.join(run_root, folder),
        )
        link(
            os.path.join(args.coco_root, "val2017", folder),
            os.path.join(run_root, "val", folder),
        )
    shutil.copyfile(
        os.path.join(args.coco_root, "train2017", "probes", args.probe + ".txt"),
        os.path.join(run_root, "train.txt"),
    )
    with open(os.path.join(run_root, "val.txt"), "w") as file:
        for im_name in sorted(
            os.listdir(os.path.join(args.coco_root, "val2017", "images"))
        ):
            file.write(f"./val/images/{im_name}\n")
    with open(args.annotations) as file:
        names = [category["name"] for category in json.load(file)["categories"]]
    write_configs(
        run_root,
        args.probe,
        names,
        "train.txt",
        "val.txt",
        COCO_TRAINING,
        args.runs_dir,
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="dataset", required=True)
    kitti = sub.add_parser("kitti", help="KITTI AL selection")
    kitti.add_argument(
        "--yolo_root",
        required=True,
        help="KITTI in YOLO format, created by yolo/kitti_to_yolo.py",
    )
    kitti.add_argument(
        "--index_file",
        required=True,
        help="AL index file, e.g. KITTI_train_indices_entropy_1_V0.txt",
    )
    kitti.add_argument("--name", required=True, help="Run name, e.g. entropy_1_V0")
    coco = sub.add_parser("coco", help="MS COCO composition probe")
    coco.add_argument(
        "--coco_root",
        required=True,
        help="MS COCO with train2017/ and val2017/ in YOLO format, created by coco_probes/coco_to_yolo.py",
    )
    coco.add_argument("--probe", required=True, help="Probe name, e.g. coco_random")
    coco.add_argument(
        "--annotations",
        required=True,
        help="instances_val2017.json, for the class names",
    )
    for p in (kitti, coco):
        p.add_argument(
            "--output_dir", required=True, help="Folder for the per-run dataset roots"
        )
        p.add_argument(
            "--runs_dir",
            required=True,
            help="Folder for the Ultralytics training outputs",
        )
    args = parser.parse_args()
    if args.dataset == "kitti":
        prepare_kitti(args)
    else:
        prepare_coco(args)


if __name__ == "__main__":
    main()
