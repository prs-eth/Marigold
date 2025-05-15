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

import io
import numpy as np
import os
import random
import tarfile
import torch
from enum import Enum
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, get_worker_info
from torchvision.transforms import InterpolationMode, Resize, ColorJitter


class DatasetMode(Enum):
    RGB_ONLY = "rgb_only"
    EVAL = "evaluate"
    TRAIN = "train"


class BaseNormalsDataset(Dataset):
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
            if os.path.isfile(dataset_dir) and tarfile.is_tarfile(dataset_dir)
            else False
        )

        if self.is_tar:
            self.tar_obj = tarfile.open(self.dataset_dir)

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
        rgb_rel_path, normals_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Normals data
        if DatasetMode.RGB_ONLY != self.mode:
            normals_data = self._load_normals_data(normals_rel_path=normals_rel_path)
            rasters.update(normals_data)

        other = {"index": index, "rgb_relative_path": rgb_rel_path}

        return rasters, other

    def _load_rgb_data(self, rgb_rel_path):
        # Read RGB data
        rgb = self._read_rgb_file(rgb_rel_path)
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

        outputs = {
            "rgb_int": torch.from_numpy(rgb).int(),
            "rgb_norm": torch.from_numpy(rgb_norm).float(),
        }
        return outputs

    def _load_normals_data(self, normals_rel_path):
        outputs = {}
        normals = torch.from_numpy(
            self._read_normals_file(normals_rel_path)
        ).float()  # [3,H,W]
        outputs["normals"] = normals

        return outputs

    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data path
        rgb_rel_path = filename_line[0]
        normals_rel_path = filename_line[1]

        return rgb_rel_path, normals_rel_path

    def _read_image(self, img_rel_path) -> np.ndarray:
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            image_to_read = self.tar_obj.extractfile("./" + img_rel_path)
            image_to_read = image_to_read.read()
            image_to_read = io.BytesIO(image_to_read)
        else:
            image_to_read = os.path.join(self.dataset_dir, img_rel_path)
        image = Image.open(image_to_read)  # [H, W, rgb]
        image = np.asarray(image)
        return image

    def _read_rgb_file(self, rel_path) -> np.ndarray:
        rgb = self._read_image(rel_path)
        rgb = np.transpose(rgb, (2, 0, 1)).astype(int)  # [rgb, H, W]
        return rgb

    def _read_normals_file(self, rel_path):
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            # normal = self.tar_obj.extractfile(f'./{tar_name}/'+rel_path)
            normal = self.tar_obj.extractfile("./" + rel_path)
            normal = normal.read()
            normal = np.load(io.BytesIO(normal))  # [H, W, 3]
        else:
            normal_path = os.path.join(self.dataset_dir, rel_path)
            normal = np.load(normal_path)
        normal = np.transpose(normal, (2, 0, 1))  # [3, H, W]
        return normal

    def _training_preprocess(self, rasters):
        # Augmentation
        if self.augm_args is not None:
            rasters = self._augment_data(rasters)

        # Resize
        if self.resize_to_hw is not None:
            resize_transform = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.BILINEAR
            )
            rasters = {k: resize_transform(v) for k, v in rasters.items()}

        return rasters

    def _augment_data(self, rasters):
        # horizontal flip (gt normals have to be flipped too)
        if random.random() < self.augm_args.lr_flip_p:
            rasters = {k: v.flip(-1) for k, v in rasters.items()}
            rasters["normals"][0, :, :] *= -1

        # if the process is on the main thread, we can use gpu to to the augmentation
        use_gpu = get_worker_info() is None
        if use_gpu:
            rasters = {k: v.cuda() for k, v in rasters.items()}

        # random gaussian blur
        if (
            random.random() < self.augm_args.gaussian_blur_p
            and rasters["rgb_int"].shape[-2] == 768
        ):  # only blur if Hypersim sample
            random_rgb_sigma = random.uniform(0.0, self.augm_args.gaussian_blur_sigma)
            rasters["rgb_int"] = TF.gaussian_blur(
                rasters["rgb_int"], kernel_size=33, sigma=random_rgb_sigma
            ).int()

        # motion blur
        if (
            random.random() < self.augm_args.motion_blur_p
            and rasters["rgb_int"].shape[-2] == 768
        ):  # only blur if Hypersim sample
            random_kernel_size = random.choice(
                [
                    x
                    for x in range(3, self.augm_args.motion_blur_kernel_size + 1)
                    if x % 2 == 1
                ]
            )
            kernel = torch.zeros(
                random_kernel_size,
                random_kernel_size,
                dtype=rasters["rgb_int"].dtype,
                device=rasters["rgb_int"].device,
            )
            kernel[random_kernel_size // 2, :] = torch.ones(random_kernel_size)
            kernel = TF.rotate(
                kernel.unsqueeze(0),
                random.uniform(0.0, self.augm_args.motion_blur_angle_range),
            )
            kernel = kernel / kernel.sum()
            channels = rasters["rgb_int"].shape[0]
            kernel = kernel.expand(channels, 1, random_kernel_size, random_kernel_size)
            rasters["rgb_int"] = (
                torch.conv2d(
                    rasters["rgb_int"].unsqueeze(0).float(),
                    kernel,
                    stride=1,
                    padding=random_kernel_size // 2,
                    groups=channels,
                )
                .squeeze(0)
                .int()
            )
        # color jitter
        if random.random() < self.augm_args.color_jitter_p:
            color_jitter = ColorJitter(
                brightness=self.augm_args.jitter_brightness_factor,
                contrast=self.augm_args.jitter_contrast_factor,
                saturation=self.augm_args.jitter_saturation_factor,
                hue=self.augm_args.jitter_hue_factor,
            )
            rgb_int_temp = rasters["rgb_int"].float() / 255.0
            rgb_int_temp = color_jitter(rgb_int_temp)
            rasters["rgb_int"] = (rgb_int_temp * 255.0).int()

        # update normalized rgb
        rasters["rgb_norm"] = rasters["rgb_int"].float() / 255.0 * 2.0 - 1.0
        return rasters

    def __del__(self):
        if hasattr(self, "tar_obj") and self.tar_obj is not None:
            self.tar_obj.close()
            self.tar_obj = None
