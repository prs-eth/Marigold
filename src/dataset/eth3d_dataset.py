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
import os
import tarfile
import torch

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode


class ETH3DDepthDataset(BaseDepthDataset):
    HEIGHT, WIDTH = 4032, 6048

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # ETH3D data parameter
            min_depth=1e-5,
            max_depth=torch.inf,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        # Read special binary data: https://www.eth3d.net/documentation#format-of-multi-view-data-image-formats
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            binary_data = self.tar_obj.extractfile("./" + rel_path)
            binary_data = binary_data.read()

        else:
            depth_path = os.path.join(self.dataset_dir, rel_path)
            with open(depth_path, "rb") as file:
                binary_data = file.read()
        # Convert the binary data to a numpy array of 32-bit floats
        depth_decoded = np.frombuffer(binary_data, dtype=np.float32).copy()

        depth_decoded[depth_decoded == torch.inf] = 0.0

        depth_decoded = depth_decoded.reshape((self.HEIGHT, self.WIDTH))
        return depth_decoded
