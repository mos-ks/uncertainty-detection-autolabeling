#!/bin/bash
# Trains YOLOv3 on the TFRecord of a KITTI AL selection and evaluates it on the KITTI validation set.
# Usage (from the repository root): bash yolo/yolov3/run_kitti_al.sh <AL TFRecord> <run name>
#   e.g. bash yolo/yolov3/run_kitti_al.sh datasets/KITTI/tf_active/entropy/V0/_train_1_V0.tfrecord entropy_1_V0
set -e
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
YOLO_DIR="$REPO_DIR/yolo/yolov3"
TFRECORD="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
MODEL_NAME="$2"

mkdir -p "$YOLO_DIR/models/$MODEL_NAME"
cd "$YOLO_DIR"
python train.py \
    --val_dataset "$REPO_DIR/datasets/KITTI/tf/_val.tfrecord" \
    --dataset "$TFRECORD" \
    --classes data/kitti.names \
    --model_name "$MODEL_NAME" \
    --num_classes 7 \
    --weights checkpoints/yolov3.tf \
    --weights_num_classes 80 \
    --transfer no_output \
    --batch_size 8 \
    --epochs 50 >> "models/$MODEL_NAME/train.log" 2>&1

python eval.py \
    --val_image_dir "$REPO_DIR/datasets/KITTI/training/image_2" \
    --val_json "$REPO_DIR/datasets/KITTI/output_coco.json" \
    --weights "models/$MODEL_NAME/yolov3_train_50.tf" \
    --classes_file data/kitti.names \
    --num_classes 7 \
    --img_size 1024 >> eval_kitti_AL.log 2>&1
