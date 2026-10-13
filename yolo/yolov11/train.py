# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
"""Trains YOLOv11 with Ultralytics from a YAML file of trainer overrides, e.g. written by prepare_run.py"""

import argparse

import yaml
from ultralytics import settings


def main(config_path):
    settings.update({"mlflow": False})
    with open(config_path, "r") as file:
        overrides = yaml.safe_load(file)
    print("### Overrides:")
    for key, value in overrides.items():
        print(f"{key}: {value}")
    # Import after updating the settings so that they take effect
    from ultralytics.models.yolo.detect import DetectionTrainer

    DetectionTrainer(overrides=overrides).train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="YAML file with the trainer overrides"
    )
    main(parser.parse_args().config)
