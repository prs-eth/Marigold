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
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import numpy as np
import random
import torch
import logging


def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_seed_sequence(
    initial_seed: int,
    length: int,
    min_val=-0x8000_0000_0000_0000,
    max_val=0xFFFF_FFFF_FFFF_FFFF,
):
    if initial_seed is None:
        logging.warning("initial_seed is None, reproducibility is not guaranteed")
    random.seed(initial_seed)

    seed_sequence = []

    for _ in range(length):
        seed = random.randint(min_val, max_val)

        seed_sequence.append(seed)

    return seed_sequence
