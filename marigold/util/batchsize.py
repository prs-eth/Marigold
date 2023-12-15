# Author: Bingxin Ke
# Last modified: 2023-12-15

import torch
import math


# Search table for suggested max. inference batch size
bs_search_table = [
    # tested on A100-PCIE-80GB
    {"res": 768, "total_vram": 79, "bs": 35},
    {"res": 1024, "total_vram": 79, "bs": 20},
    # tested on A100-PCIE-40GB
    {"res": 768, "total_vram": 39, "bs": 15},
    {"res": 1024, "total_vram": 39, "bs": 8},
    # tested on RTX3090, RTX4090
    {"res": 512, "total_vram": 23, "bs": 20},
    {"res": 768, "total_vram": 23, "bs": 7},
    {"res": 1024, "total_vram": 23, "bs": 3},
    # tested on GTX1080Ti
    {"res": 512, "total_vram": 10, "bs": 5},
    {"res": 768, "total_vram": 10, "bs": 2},
]


def find_batch_size(ensemble_size: int, input_res: int) -> int:
    """
    Automatically search for suitable operating batch size.

    Args:
        ensemble_size (int): Number of predictions to be ensembled
        input_res (int): Operating resolution of the input image.

    Returns:
        int: Operating batch size
    """
    if not torch.cuda.is_available():
        return 1

    total_vram = torch.cuda.mem_get_info()[1] / 1024.0**3

    for settings in sorted(bs_search_table, key=lambda k: (k["res"], -k["total_vram"])):
        if input_res <= settings["res"] and total_vram >= settings["total_vram"]:
            bs = settings["bs"]
            if bs > ensemble_size:
                bs = ensemble_size
            elif bs > math.ceil(ensemble_size / 2) and bs < ensemble_size:
                bs = math.ceil(ensemble_size / 2)
            return bs

    return 1
