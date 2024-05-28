# Last modified: 2024-04-16
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

import os

from .base_depth_dataset import BaseDepthDataset, get_pred_name, DatasetMode  # noqa: F401
from .diode_dataset import DIODEDataset
from .eth3d_dataset import ETH3DDataset
from .hypersim_dataset import HypersimDataset
from .kitti_dataset import KITTIDataset
from .nyu_dataset import NYUDataset
from .scannet_dataset import ScanNetDataset
from .vkitti_dataset import VirtualKITTIDataset


dataset_name_class_dict = {
    "hypersim": HypersimDataset,
    "vkitti": VirtualKITTIDataset,
    "nyu_v2": NYUDataset,
    "kitti": KITTIDataset,
    "eth3d": ETH3DDataset,
    "diode": DIODEDataset,
    "scannet": ScanNetDataset,
}


def get_dataset(
    cfg_data_split, base_data_dir: str, mode: DatasetMode, **kwargs
) -> BaseDepthDataset:
    if "mixed" == cfg_data_split.name:
        assert DatasetMode.TRAIN == mode, "Only training mode supports mixed datasets."
        dataset_ls = [
            get_dataset(_cfg, base_data_dir, mode, **kwargs)
            for _cfg in cfg_data_split.dataset_list
        ]
        return dataset_ls
    elif cfg_data_split.name in dataset_name_class_dict.keys():
        dataset_class = dataset_name_class_dict[cfg_data_split.name]
        dataset = dataset_class(
            mode=mode,
            filename_ls_path=cfg_data_split.filenames,
            dataset_dir=os.path.join(base_data_dir, cfg_data_split.dir),
            **cfg_data_split,
            **kwargs,
        )
    else:
        raise NotImplementedError

    return dataset
