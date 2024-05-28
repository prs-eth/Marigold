# Last modified: 2024-02-08
#
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
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
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import torch
import tarfile
import os
import numpy as np

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode


class ETH3DDataset(BaseDepthDataset):
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
