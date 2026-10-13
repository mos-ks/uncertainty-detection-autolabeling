# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
"""Core-set selection via greedy k-center on whole-image DINOv2 embeddings"""

import argparse
import ast
import os

import cv2
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def extract_embeddings(image_dir, im_names, batch_size=64):
    """Extracts the DINOv2 ViT-S/14 embedding of each whole image"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(device)
    embeddings, batch = [], []

    def flush():
        if batch:
            with torch.no_grad():
                embeddings.append(
                    model(torch.stack(batch).to(device)).float().cpu().numpy()
                )
            batch.clear()

    for i, im_name in enumerate(im_names):
        rgb = cv2.cvtColor(
            cv2.imread(os.path.join(image_dir, im_name)), cv2.COLOR_BGR2RGB
        )
        im = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
        batch.append(torch.from_numpy(((im - MEAN) / STD).transpose(2, 0, 1)))
        if len(batch) >= batch_size:
            flush()
        if (i + 1) % 1000 == 0:
            print(f"Embedded {i + 1}/{len(im_names)} images")
    flush()
    return np.concatenate(embeddings, 0)


def greedy_kcenter(embeddings, labeled_idx, target_n):
    """Grows the labeled set to target_n by repeatedly adding the image with the largest cosine distance to its nearest labeled image"""
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    labeled = np.zeros(len(embeddings), bool)
    labeled[list(labeled_idx)] = True
    min_dist = 1.0 - (normed @ normed[labeled].T).max(axis=1)
    min_dist[labeled] = -np.inf
    while labeled.sum() < target_n:
        j = int(np.argmax(min_dist))
        labeled[j] = True
        min_dist[j] = -np.inf
        np.minimum(min_dist, 1.0 - (normed @ normed[j]), out=min_dist, where=~labeled)
        min_dist[labeled] = -np.inf
    return np.where(labeled)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--general_path",
        type=str,
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="Path to working space",
    )
    parser.add_argument(
        "--warmup_file",
        type=str,
        required=True,
        help="Index file of the common warm-up set, e.g. datasets/KITTI/tf_active/random/V0/KITTI_train_indices_random_0_V0.txt",
    )
    parser.add_argument(
        "--iteration_budget",
        type=str,
        default="[5, 5, 5, 10, 20, 30, 25]",
        help="Iteration budget in percent, as in active_learning_loop.py",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=3,
        help="Number of AL iterations after the warm-up",
    )
    parser.add_argument(
        "--added_name",
        type=str,
        default="_V0",
        help="Suffix of the index files, as in active_learning_loop.py",
    )
    args = parser.parse_args()

    general_path = args.general_path + "/"
    image_dir = general_path + "datasets/KITTI/training/image_2/"
    im_names = sorted(os.listdir(image_dir))
    with open(general_path + "datasets/KITTI/vaL_index_list.txt") as file:
        val_indices = {int(line.strip()) for line in file if line.strip()}
    available_indices = [i for i in range(len(im_names)) if i not in val_indices]

    iteration_budget = ast.literal_eval(args.iteration_budget)
    ims_per_iter = [int(len(available_indices) * b / 100) for b in iteration_budget]

    embeddings_path = (
        general_path + "datasets/KITTI/training/dinov2_vits14_embeddings.npy"
    )
    if os.path.exists(embeddings_path):
        embeddings = np.load(embeddings_path)
    else:
        embeddings = extract_embeddings(image_dir, im_names)
        np.save(embeddings_path, embeddings)
    pool_embeddings = embeddings[available_indices]
    pool_row = {idx: row for row, idx in enumerate(available_indices)}

    with open(args.warmup_file) as file:
        selected = {pool_row[int(line.strip())] for line in file if line.strip()}

    output_path = (
        general_path
        + "datasets/KITTI/tf_active/coreset_dino/"
        + args.added_name.split("_")[-1]
        + "/"
    )
    os.makedirs(output_path, exist_ok=True)
    for iteration in range(args.num_iterations + 1):
        if iteration > 0:
            selected = set(
                greedy_kcenter(
                    pool_embeddings, selected, sum(ims_per_iter[: iteration + 1])
                ).tolist()
            )
        index_file = (
            output_path
            + f"KITTI_train_indices_coreset_dino_{iteration}{args.added_name}.txt"
        )
        with open(index_file, "w") as file:
            for row in sorted(selected):
                file.write(f"{available_indices[row]}\n")
        print(f"Iteration {iteration}: {len(selected)} images -> {index_file}")


if __name__ == "__main__":
    main()
