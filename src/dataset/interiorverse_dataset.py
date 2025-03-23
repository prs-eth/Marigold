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

import numpy as np
import torch

from src.util.image_util import img_linear2srgb, is_hdr
from .base_iid_dataset import BaseIIDDataset, DatasetMode
from .base_normals_dataset import BaseNormalsDataset


class InteriorVerseNormalsDataset(BaseNormalsDataset):
    pass


# https://github.com/jingsenzhu/IndoorInverseRendering/tree/main/interiorverse
class InteriorVerseIIDDataset(BaseIIDDataset):
    def _load_targets_data(self, rel_paths):
        albedo_path = rel_paths[0]  # 000_albedo.exr
        material_path = rel_paths[1]  # 000_material.exr
        mask_path = rel_paths[2]  # 000_mask.exr

        albedo_img = self._read_image(albedo_path)
        material_img = self._read_image(
            material_path
        )  # R is roughness, G is metallicity
        material_img[2, :, :] = 0

        mask_img = self._read_image(mask_path)
        mask_img = mask_img != 0  # Convert to a boolean np array
        mask_img_squeezed = np.expand_dims(
            np.all(mask_img, axis=0), axis=0
        )  # Convert 3 channel to 1 channel

        # SD is pretrained in sRGB space. If we load HDR data, we should also convert to sRGB space.
        if is_hdr(albedo_path):
            albedo_img = img_linear2srgb(albedo_img)

        if is_hdr(material_path):
            material_img = img_linear2srgb(material_img)

        outputs = {
            "albedo": torch.from_numpy(albedo_img),
            "material": torch.from_numpy(material_img),
            "mask": torch.from_numpy(mask_img_squeezed),
        }

        # add three channel mask for evaluation
        if self.mode == DatasetMode.EVAL:
            eval_masks = {
                "mask_albedo": torch.from_numpy(mask_img).bool(),
                "mask_material": torch.from_numpy(mask_img).bool(),
            }
            outputs.update(eval_masks)

        return outputs
