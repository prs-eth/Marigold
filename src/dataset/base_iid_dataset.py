# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
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
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # Enable OpenCV support for EXR
# ruff: noqa: E402

import io
import tarfile
import numpy as np
import random
import torch
from enum import Enum
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize

from src.util.image_util import (
    img_hwc2chw,
    img_linear2srgb,
    is_hdr,
    read_img_from_file,
    read_img_from_tar,
)


class DatasetMode(Enum):
    RGB_ONLY = "rgb_only"
    EVAL = "evaluate"
    TRAIN = "train"


class BaseIIDDataset(Dataset):
    def __init__(
        self,
        mode: DatasetMode,
        filename_ls_path: str,
        dataset_dir: str,
        disp_name: str,
        augmentation_args: dict = None,
        resize_to_hw=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        # dataset info
        self.filename_ls_path = filename_ls_path
        self.dataset_dir = dataset_dir
        assert os.path.exists(
            self.dataset_dir
        ), f"Dataset does not exist at: {self.dataset_dir}"
        self.disp_name = disp_name

        # training arguments
        self.augm_args = augmentation_args
        self.resize_to_hw = resize_to_hw

        # Load filenames
        with open(self.filename_ls_path, "r") as f:
            self.filenames = [s.split() for s in f.readlines()]

        # Tar dataset
        self.tar_obj = None
        self.is_tar = (
            True
            if os.path.isfile(self.dataset_dir) and tarfile.is_tarfile(self.dataset_dir)
            else False
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        rasters, other = self._get_data_item(index)
        if DatasetMode.TRAIN == self.mode:
            rasters = self._training_preprocess(rasters)
        # merge
        outputs = rasters
        outputs.update(other)
        return outputs

    def _get_data_item(self, index):
        rgb_rel_path, targets_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # load targets data should be filled in specialized dataset definition
        if DatasetMode.RGB_ONLY != self.mode:
            targets_data = self._load_targets_data(rel_paths=targets_rel_path)
            rasters.update(targets_data)

        other = {"index": index, "rgb_relative_path": rgb_rel_path}

        return rasters, other

    def _read_image(self, rel_path) -> np.ndarray:
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            img = read_img_from_tar(self.tar_obj, rel_path)
        else:
            img = read_img_from_file(os.path.join(self.dataset_dir, rel_path))

        if len(img.shape) == 3:  # hwc->chw, except for single-channel images
            img = img_hwc2chw(img)

        assert img.min() >= 0 and img.max() <= 1
        return img

    def _load_rgb_data(self, rgb_rel_path):
        # rgb is in [0,1] range
        rgb = self._read_image(rgb_rel_path)

        # SD is pretrained in sRGB space. If we load HDR data, we should also convert to sRGB space.
        if is_hdr(rgb_rel_path):
            rgb = img_linear2srgb(rgb)

        outputs = {"rgb": torch.from_numpy(rgb).float()}  # [0,1]

        return outputs

    def _read_numpy(self, rel_path):
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            image_to_read = self.tar_obj.extractfile("./" + rel_path)
            image_to_read = image_to_read.read()
            image_to_read = io.BytesIO(image_to_read)
        else:
            image_to_read = os.path.join(self.dataset_dir, rel_path)

        image = np.load(image_to_read).transpose((2, 0, 1))  # [3,H,W]
        return image

    def _load_targets_data(self, rel_paths):
        outputs = {}
        return outputs

    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Only the first is the input image, the rest should be specialized by each dataset.
        rgb_rel_path = filename_line[0]
        targets_rel_path = filename_line[1:]

        return rgb_rel_path, targets_rel_path

    def _training_preprocess(self, rasters):
        # Augmentation
        if self.augm_args is not None:
            rasters = self._augment_data(rasters)

        # Resize
        if self.resize_to_hw is not None:
            resize_bilinear = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.BILINEAR
            )
            resize_nearest = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT
            )

            rasters = {
                k: (resize_nearest(v) if "valid_mask" in k else resize_bilinear(v))
                for k, v in rasters.items()
            }

        return rasters

    def _augment_data(self, rasters):
        # horizontal flip
        if random.random() < self.augm_args.lr_flip_p:
            rasters = {k: v.flip(-1) for k, v in rasters.items()}

    def __del__(self):
        if hasattr(self, "tar_obj") and self.tar_obj is not None:
            self.tar_obj.close()
            self.tar_obj = None
