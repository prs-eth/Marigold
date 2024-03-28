# Author: Bingxin Ke
# Last modified: 2024-02-08

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
