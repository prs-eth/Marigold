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

from .base_normals_dataset import BaseNormalsDataset

# Sintel original resolution
H, W = 436, 1024


# crop to [436,582] --> later upsample with factor 1.1 to [480,640]
# crop off 221 pixels on both sides (1024 - 2*221 = 582)
def center_crop(img):
    assert img.shape[0] == 3 or img.shape[0] == 1, "Channel dim should be first dim"
    crop = 221

    out = img[:, :, crop : W - crop]  # [3,H,W]

    return out  # [3,436,582]


class SintelNormalsDataset(BaseNormalsDataset):
    def _load_rgb_data(self, rgb_rel_path):
        # Read RGB data
        rgb = self._read_rgb_file(rgb_rel_path)
        rgb = center_crop(rgb)
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

        # replace invalid sky values with camera facing normals
        valid_normal_mask = torch.norm(normals, p=2, dim=0) > 0.1
        normals[:, ~valid_normal_mask] = torch.tensor(
            [0.0, 0.0, 1.0], dtype=normals.dtype
        ).view(3, 1)
        # crop on both sides
        outputs["normals"] = center_crop(normals)

        return outputs
