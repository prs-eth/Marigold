# Author: Bingxin Ke
# Last modified: 2024-02-22

import os


def is_on_slurm():
    cluster_name = os.getenv("SLURM_CLUSTER_NAME")
    is_on_slurm = cluster_name is not None
    return is_on_slurm


def get_local_scratch_dir():
    local_scratch_dir = os.getenv("TMPDIR")
    return local_scratch_dir
