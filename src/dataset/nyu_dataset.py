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


class NYUDepthDataset(BaseDepthDataset):
    def __init__(
        self,
        eigen_valid_mask: bool,
        **kwargs,
    ) -> None:
        super().__init__(
            # NYUv2 dataset parameter
            min_depth=1e-3,
            max_depth=10.0,
            has_filled_depth=True,
            name_mode=DepthFileNameMode.rgb_id,
            **kwargs,
        )

        self.eigen_valid_mask = eigen_valid_mask

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode NYU depth
        depth_decoded = depth_in / 1000.0
        return depth_decoded

    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = super()._get_valid_mask(depth)

        # Eigen crop for evaluation
        if self.eigen_valid_mask:
            eval_mask = torch.zeros_like(valid_mask.squeeze()).bool()
            eval_mask[45:471, 41:601] = 1
            eval_mask.reshape(valid_mask.shape)
            valid_mask = torch.logical_and(valid_mask, eval_mask)

        return valid_mask


class NYUNormalsDataset(BaseNormalsDataset):
    pass
