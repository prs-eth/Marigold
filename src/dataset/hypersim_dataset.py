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

import torch

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode
from .base_normals_dataset import BaseNormalsDataset
from .base_iid_dataset import BaseIIDDataset


class HypersimDepthDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # Hypersim data parameter
            min_depth=1e-5,
            max_depth=65.0,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.rgb_i_d,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode Hypersim depth
        depth_decoded = depth_in / 1000.0
        return depth_decoded


class HypersimNormalsDataset(BaseNormalsDataset):
    pass


class HypersimIIDDataset(BaseIIDDataset):
    def _load_targets_data(self, rel_paths):
        albedo_path = rel_paths[0]  # albedo_cam_00_fr0000.npy
        shading_path = rel_paths[1]  # shading_cam_00_fr0000.npy
        residual_path = rel_paths[2]  # residual_cam_00_fr0000.npy

        albedo_raw = self._read_numpy(albedo_path)  # in linear space
        shading_raw = self._read_numpy(shading_path)
        residual_raw = self._read_numpy(residual_path)

        rasters = {
            "albedo": torch.from_numpy(albedo_raw).float(),  # [0,1] linear space
            "shading_raw": torch.from_numpy(shading_raw),
            "residual_raw": torch.from_numpy(residual_raw),
        }
        del albedo_raw, shading_raw, residual_raw

        # get the cut off value for shading/residual
        cut_off_value = self._get_cut_off_value(rasters)
        # clip and normalize shading and residual (to bring them to the same scale and value range)
        rasters = self._process_shading_residual(rasters, cut_off_value)

        # Load masks
        valid_mask_albedo, valid_mask_shading, valid_mask_residual = (
            self._get_valid_masks(rasters)
        )
        rasters.update(
            {
                "mask_albedo": valid_mask_albedo.bool(),
                "mask_shading": valid_mask_shading.bool(),
                "mask_residual": valid_mask_residual.bool(),
            }
        )
        return rasters

    def _process_shading_residual(self, rasters, cut_off_value):
        # Clip by cut_off_value
        shading_clipped = torch.clip(rasters["shading_raw"], 0, cut_off_value)
        residual_clipped = torch.clip(rasters["residual_raw"], 0, cut_off_value)
        # Divide by them same cut off value to bring them to the same scale
        shading_norm = shading_clipped / cut_off_value  #  [0,1]
        residual_norm = residual_clipped / cut_off_value  # [0,1]

        rasters.update(
            {
                "shading": shading_norm.float(),
                "residual": residual_norm.float(),
            }
        )
        return rasters

    def _get_cut_off_value(self, rasters):
        shading_raw = rasters["shading_raw"]
        residual_raw = rasters["residual_raw"]

        # take the maximum of residual_98 and shading_98 as cutoff value
        residual_98 = torch.quantile(residual_raw, 0.98)
        shading_98 = torch.quantile(shading_raw, 0.98)
        cut_off_value = torch.max(torch.tensor([residual_98, shading_98]))

        return cut_off_value

    def _get_valid_masks(self, rasters):
        albedo_gt_ts = rasters["albedo"]  # [3,H,W]
        invalid_mask_albedo = torch.isnan(albedo_gt_ts) | torch.isinf(albedo_gt_ts)
        zero_mask = (albedo_gt_ts == 0).all(dim=0, keepdim=True)
        zero_mask = zero_mask.expand_as(albedo_gt_ts)
        invalid_mask_albedo |= zero_mask
        valid_mask_albedo = ~invalid_mask_albedo

        shading_gt_ts = rasters["shading"]
        invalid_mask_shading = torch.isnan(shading_gt_ts) | torch.isinf(shading_gt_ts)
        valid_mask_shading = ~invalid_mask_shading

        residual_gt_ts = rasters["residual"]
        invalid_mask_residual = torch.isnan(residual_gt_ts) | torch.isinf(
            residual_gt_ts
        )
        valid_mask_residual = ~invalid_mask_residual

        return valid_mask_albedo, valid_mask_shading, valid_mask_residual
