# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Modified.
Hparams for model architecture and trainer."""
import ast
import collections
import copy
from typing import Any, Dict, Text

import numpy as np
import six
import tensorflow as tf
import yaml


def eval_str_fn(val):
    if val in {"true", "false"}:
        return val == "true"
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


# pylint: disable=protected-access
class Config(object):
    """A config utility class."""

    def __init__(self, config_dict=None):
        self.update(config_dict)

    def __setattr__(self, k, v):
        self.__dict__[k] = Config(v) if isinstance(v, dict) else copy.deepcopy(v)

    def __getattr__(self, k):
        return self.__dict__[k]

    def __getitem__(self, k):
        return self.__dict__[k]

    def __repr__(self):
        return repr(self.as_dict())

    def __deepcopy__(self, memodict):
        return type(self)(self.as_dict())

    def __str__(self):
        try:
            return yaml.dump(self.as_dict(), indent=4)
        except TypeError:
            return str(self.as_dict())

    def _update(self, config_dict, allow_new_keys=True):
        """Recursively update internal members."""
        if not config_dict:
            return

        for k, v in six.iteritems(config_dict):
            if k not in self.__dict__:
                if allow_new_keys:
                    self.__setattr__(k, v)
                else:
                    raise KeyError("Key `{}` does not exist for overriding. ".format(k))
            else:
                if isinstance(self.__dict__[k], Config) and isinstance(v, dict):
                    self.__dict__[k]._update(v, allow_new_keys)
                elif isinstance(self.__dict__[k], Config) and isinstance(v, Config):
                    self.__dict__[k]._update(v.as_dict(), allow_new_keys)
                else:
                    self.__setattr__(k, v)

    def get(self, k, default_value=None):
        return self.__dict__.get(k, default_value)

    def update(self, config_dict):
        """Update members while allowing new keys."""
        self._update(config_dict, allow_new_keys=True)

    def keys(self):
        return self.__dict__.keys()

    def override(self, config_dict_or_str, allow_new_keys=False):
        """Update members while disallowing new keys."""
        if isinstance(config_dict_or_str, str):
            if not config_dict_or_str:
                return
            elif "=" in config_dict_or_str:
                config_dict = self.parse_from_str(config_dict_or_str)
            elif config_dict_or_str.endswith(".yaml"):
                config_dict = self.parse_from_yaml(config_dict_or_str)
            else:
                raise ValueError(
                    'Invalid string {}, must end with .yaml or contains "=".'.format(
                        config_dict_or_str
                    )
                )
        elif isinstance(config_dict_or_str, dict):
            config_dict = config_dict_or_str
        else:
            raise ValueError("Unknown value type: {}".format(config_dict_or_str))

        self._update(config_dict, allow_new_keys)

    def parse_from_yaml(self, yaml_file_path: Text) -> Dict[Any, Any]:
        """Parses a yaml file and returns a dictionary."""
        with tf.io.gfile.GFile(yaml_file_path, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
            return config_dict

    def save_to_yaml(self, yaml_file_path):
        """Write a dictionary into a yaml file."""
        with tf.io.gfile.GFile(yaml_file_path, "w") as f:
            yaml.dump(self.as_dict(), f, default_flow_style=False)

    def parse_from_str(self, config_str: Text) -> Dict[Any, Any]:
        """Parse a string like 'x.y=1,x.z=2' to nested dict {x: {y: 1, z: 2}}."""
        if not config_str:
            return {}
        config_dict = {}
        try:
            for kv_pair in config_str.split(","):
                if not kv_pair:  # skip empty string
                    continue
                key_str, value_str = kv_pair.split("=")
                key_str = key_str.strip()

                def add_kv_recursive(k, v):
                    """Recursively parse x.y.z=tt to {x: {y: {z: tt}}}."""
                    if "." not in k:
                        if "*" in v:
                            # we reserve * to split arrays.
                            return {k: [eval_str_fn(vv) for vv in v.split("*")]}
                        return {k: eval_str_fn(v)}
                    pos = k.index(".")
                    return {k[:pos]: add_kv_recursive(k[pos + 1 :], v)}

                def merge_dict_recursive(target, src):
                    """Recursively merge two nested dictionary."""
                    for k in src.keys():
                        if (
                            k in target
                            and isinstance(target[k], dict)
                            and isinstance(src[k], collections.Mapping)
                        ):
                            merge_dict_recursive(target[k], src[k])
                        else:
                            target[k] = src[k]

                merge_dict_recursive(config_dict, add_kv_recursive(key_str, value_str))
            return config_dict
        except ValueError:
            raise ValueError("Invalid config_str: {}".format(config_str))

    def as_dict(self):
        """Returns a dict representation."""
        config_dict = {}
        for k, v in six.iteritems(self.__dict__):
            if isinstance(v, Config):
                config_dict[k] = v.as_dict()
            else:
                config_dict[k] = copy.deepcopy(v)
        return config_dict
        # pylint: enable=protected-access


def default_detection_configs():
    """Returns a default detection configs."""
    h = Config()

    # Custom parameters
    h.early_stopping_patience = 0

    h.infer_draw_uncert = (
        True  # Save an image with the uncertainty for each predicted image
    )
    h.loss_attenuation = False  # Estimate box aleatoric uncertaianty
    h.clip_min_uncert = 0.01  # Clip uncertainty during training if values exceed
    h.clip_max_uncert = (
        1024  # Clip uncertainty during training if values exceed (input image size)
    )
    h.uncert_adjust_method = "l-norm"  # Decoding method for box uncertainty [l-norm, n-flow, falsedec, sample]
    h.decode_nsamples = 100  # If using sampling for decoding, define number of samples

    h.mc_dropout = False  # Activate MC dropout
    h.mc_dropoutrate = 0.0  # Setting it while headrates=0 will activate full MC dropout
    h.mc_classheadrate = (
        0.0  # Setting it while mcdropoutrate=0 will activate only for class head
    )
    h.mc_boxheadrate = (
        0.0  # Setting it while mcdropoutrate=0 will activate only for box head
    )
    h.mc_dropoutsamp = 10  # Number of dropout iterations during inference

    h.assign_gt_box = "IoU"  # ["MSE","IoU",False] Reorder NMS output for calibration and validation based on MSE/IoU instead of classification score

    h.enable_softmax = False  # Enable the propagation of the logits to the output
    h.calibrate_classification = True  # Calibrate classification logits
    h.calib_method_class = (
        "iso_percls"  # ["ts_all", "ts_percls", "iso_all", "iso_percls"]
    )
    h.calibrate_regression = True  # Calibrate regression uncertainty
    h.calib_method_box = "iso_perclscoo"  # ["ts_all", "ts_percoo", "iso_all", "iso_percoo", "iso_perclscoo", "rel_iso_perclscoo"]

    h.count_classes = (
        False  # Prints and plots class distribution for the dataset before training
    )
    h.boxloss_type = "huber"  # Box loss type for trianing ["MSE", "huber"]
    h.save_freq = 1  # Epoch frequency of saving the model
    h.sample_images = None  # List of images path to test the model on during training
    h.sample_images_freq = None  # Epochs of how often to test the model on the images
    h.save_train_images = (
        False  # Save input images as they are fed to the model to tensorboard
    )
    h.autoaugment_policy = None  # Augmentation method; 'v0' for autoaug, 'randaug' for randaug and 'albu' for albumentations
    h.albumentations_mode = "optimal"  # Augmentation parameters selection method; 'random', 'optimal' or 'subjective'
    h.albumentations_path = "../configs/augmentation/"  # Path to read optimal parameters from; adds to it, e.g., rain/rain_opt_params
    h.albumentations_ops = [
        "rain",
        "snow",
        "fog",
        "sat",
    ]  # "brct", "eql" # Augmentation operations
    h.consistency_ssl = False  # Performs real-time augmentation during inference and saves mIoU and class agreement into txt
    h.infer_augment = False  # Perform synthetic augmentations during inference, ["heq", "alb", "aug", "flip"]

    h.thr_fpr_tpr = 0.95  # Budget for thresholding
    h.thr_cd = True  # Select budget on correct/missing detections
    h.thr_iou_thrs = list(
        np.round(np.arange(0.50, 0.76, 0.05), 2)
    )  # Thresholding ious used for optimization and metrics calculation, default to np.arange(0.5,0.76,0.05)
    h.thr_sel_uncert = "ENTALBOX"  # Defaults to ENTALBOX. Select which uncertainties to use for thresholding, in this case entropy and aleatoric localization uncertainty

    # model name.
    h.name = "efficientdet-d1"

    # activation type: see activation_fn in utils.py.
    h.act_type = "swish"

    # input preprocessing parameters
    h.image_size = 640  # An integer or a string WxH such as 640x320.
    h.target_size = None
    h.input_rand_hflip = True
    h.jitter_min = 0.1
    h.jitter_max = 2.0
    h.grid_mask = False
    h.map_freq = 5  # AP eval frequency in epochs.

    # dataset specific parameters
    # TODO(tanmingxing): update this to be 91 for COCO, and 21 for pascal.
    h.num_classes = 90  # 1+ actual classes, 0 is reserved for background.
    h.seg_num_classes = 3  # segmentation classes
    h.heads = ["object_detection"]  # 'object_detection', 'segmentation'

    h.skip_crowd_during_training = True
    h.label_map = None  # a dict or a string of 'coco', 'voc', 'waymo'.
    h.max_instances_per_image = 100  # Default to 100 for COCO.
    h.regenerate_source_id = False

    # model architecture
    h.min_level = 3
    h.max_level = 7
    h.num_scales = 3
    # ratio w/h: 2.0 means w=1.4, h=0.7. Can be computed with k-mean per dataset.
    h.aspect_ratios = [1.0, 2.0, 0.5]  # [[0.7, 1.4], [1.0, 1.0], [1.4, 0.7]] #
    h.anchor_scale = 4.0
    # is batchnorm training mode
    h.is_training_bn = True
    # optimization
    h.momentum = 0.9
    h.optimizer = "sgd"  # can be 'adam' or 'sgd'.
    h.learning_rate = 0.08  # 0.008 for adam.
    h.lr_warmup_init = 0.008  # 0.0008 for adam.
    h.lr_warmup_epoch = 1.0
    h.first_lr_drop_epoch = 200.0
    h.second_lr_drop_epoch = 250.0
    h.poly_lr_power = 0.9
    h.clip_gradients_norm = 10.0
    h.num_epochs = 300
    h.data_format = "channels_last"
    # The default image normalization is identical to Cloud TPU ResNet.
    h.mean_rgb = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    h.stddev_rgb = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    h.scale_range = False

    # classification loss
    h.label_smoothing = 0.0  # 0.1 is a good default
    # Behold the focal loss parameters
    h.alpha = 0.25
    h.gamma = 1.5

    # localization loss
    h.delta = 0.1  # regularization parameter of huber loss.
    # total loss = box_loss * box_loss_weight + iou_loss * iou_loss_weight
    h.box_loss_weight = 50.0
    h.iou_loss_type = None
    h.iou_loss_weight = 1.0

    # regularization l2 loss.
    h.weight_decay = 4e-5
    h.strategy = None  # 'tpu', 'gpus', None
    h.mixed_precision = False  # If False, use float32.
    h.loss_scale = None  # set to 2**16 enables dynamic loss scale

    # For detection.
    h.box_class_repeats = 3
    h.fpn_cell_repeats = 3
    h.fpn_num_filters = 88
    h.separable_conv = True
    h.apply_bn_for_resampling = True
    h.conv_after_downsample = False
    h.conv_bn_act_pattern = False
    h.drop_remainder = True  # drop remainder for the final batch eval.

    # For post-processing nms, must be a dict.
    h.nms_configs = {
        "method": "gaussian",
        "iou_thresh": None,  # use the default value based on method.
        "score_thresh": 0.0,
        "sigma": None,
        "pyfunc": False,
        "max_nms_inputs": 0,
        "max_output_size": 100,
    }
    h.tflite_max_detections = 100

    # version.
    h.fpn_name = None
    h.fpn_weight_method = None
    h.fpn_config = None

    # No stochastic depth in default.
    h.survival_prob = None

    h.lr_decay_method = "cosine"
    h.moving_average_decay = 0.9998
    h.ckpt_var_scope = None  # ckpt variable scope.
    # If true, skip loading pretrained weights if shape mismatches.
    h.skip_mismatch = True

    h.backbone_name = "efficientnet-b1"
    h.backbone_config = None
    h.var_freeze_expr = None

    # A temporary flag to switch between legacy and keras models.
    h.use_keras_model = True
    h.dataset_type = None
    h.positives_momentum = None
    h.grad_checkpoint = False

    # Parameters for the Checkpoint Callback.
    h.verbose = 1

    return h


efficientdet_model_param_dict = {
    "efficientdet-d0": dict(
        name="efficientdet-d0",
        backbone_name="efficientnet-b0",
        image_size=512,
        fpn_num_filters=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
    ),
    "efficientdet-d1": dict(
        name="efficientdet-d1",
        backbone_name="efficientnet-b1",
        image_size=640,
        fpn_num_filters=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
    ),
    "efficientdet-d2": dict(
        name="efficientdet-d2",
        backbone_name="efficientnet-b2",
        image_size=768,
        fpn_num_filters=112,
        fpn_cell_repeats=5,
        box_class_repeats=3,
    ),
    "efficientdet-d3": dict(
        name="efficientdet-d3",
        backbone_name="efficientnet-b3",
        image_size=896,
        fpn_num_filters=160,
        fpn_cell_repeats=6,
        box_class_repeats=4,
    ),
    "efficientdet-d4": dict(
        name="efficientdet-d4",
        backbone_name="efficientnet-b4",
        image_size=1024,
        fpn_num_filters=224,
        fpn_cell_repeats=7,
        box_class_repeats=4,
    ),
    "efficientdet-d5": dict(
        name="efficientdet-d5",
        backbone_name="efficientnet-b5",
        image_size=1280,
        fpn_num_filters=288,
        fpn_cell_repeats=7,
        box_class_repeats=4,
    ),
    "efficientdet-d6": dict(
        name="efficientdet-d6",
        backbone_name="efficientnet-b6",
        image_size=1280,
        fpn_num_filters=384,
        fpn_cell_repeats=8,
        box_class_repeats=5,
        fpn_weight_method="sum",  # Use unweighted sum for stability.
    ),
    "efficientdet-d7": dict(
        name="efficientdet-d7",
        backbone_name="efficientnet-b6",
        image_size=1536,
        fpn_num_filters=384,
        fpn_cell_repeats=8,
        box_class_repeats=5,
        anchor_scale=5.0,
        fpn_weight_method="sum",  # Use unweighted sum for stability.
    ),
    "efficientdet-d7x": dict(
        name="efficientdet-d7x",
        backbone_name="efficientnet-b7",
        image_size=1536,
        fpn_num_filters=384,
        fpn_cell_repeats=8,
        box_class_repeats=5,
        anchor_scale=4.0,
        max_level=8,
        fpn_weight_method="sum",  # Use unweighted sum for stability.
    ),
}


lite_common_param = dict(
    mean_rgb=127.0,
    stddev_rgb=128.0,
    act_type="relu6",
    fpn_weight_method="sum",
)

efficientdet_lite_param_dict = {
    # lite models are in progress and subject to changes.
    # mean_rgb and stddev_rgb are consistent with EfficientNet-Lite models in
    # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/efficientnet_lite_builder.py#L28
    "efficientdet-lite0": dict(
        name="efficientdet-lite0",
        backbone_name="efficientnet-lite0",
        image_size=320,
        fpn_num_filters=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        anchor_scale=3.0,
        **lite_common_param,
    ),
    "efficientdet-lite1": dict(
        name="efficientdet-lite1",
        backbone_name="efficientnet-lite1",
        image_size=384,
        fpn_num_filters=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        anchor_scale=3.0,
        **lite_common_param,
    ),
    "efficientdet-lite2": dict(
        name="efficientdet-lite2",
        backbone_name="efficientnet-lite2",
        image_size=448,
        fpn_num_filters=112,
        fpn_cell_repeats=5,
        box_class_repeats=3,
        anchor_scale=3.0,
        **lite_common_param,
    ),
    "efficientdet-lite3": dict(
        name="efficientdet-lite3",
        backbone_name="efficientnet-lite3",
        image_size=512,
        fpn_num_filters=160,
        fpn_cell_repeats=6,
        box_class_repeats=4,
        **lite_common_param,
    ),
    "efficientdet-lite3x": dict(
        name="efficientdet-lite3x",
        backbone_name="efficientnet-lite3",
        image_size=640,
        fpn_num_filters=200,
        fpn_cell_repeats=6,
        box_class_repeats=4,
        anchor_scale=3.0,
        **lite_common_param,
    ),
    "efficientdet-lite4": dict(
        name="efficientdet-lite4",
        backbone_name="efficientnet-lite4",
        image_size=640,
        fpn_num_filters=224,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        **lite_common_param,
    ),
}


def get_efficientdet_config(model_name="efficientdet-d1"):
    """Get the default config for EfficientDet based on model name."""
    h = default_detection_configs()
    if model_name in efficientdet_model_param_dict:
        h.override(efficientdet_model_param_dict[model_name])
    elif model_name in efficientdet_lite_param_dict:
        h.override(efficientdet_lite_param_dict[model_name])
    else:
        raise ValueError("Unknown model name: {}".format(model_name))

    return h


def get_detection_config(model_name):
    if model_name.startswith("efficientdet"):
        return get_efficientdet_config(model_name)
    else:
        raise ValueError("model name must start with efficientdet.")
