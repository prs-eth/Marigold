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

from .marigold_depth_trainer import MarigoldDepthTrainer
from .marigold_iid_trainer import MarigoldIIDTrainer
from .marigold_normals_trainer import MarigoldNormalsTrainer
from .marigold_depth_LCM_trainer import MarigoldDepthLCMTrainer


trainer_cls_name_dict = {
    "MarigoldDepthTrainer": MarigoldDepthTrainer,
    "MarigoldIIDTrainer": MarigoldIIDTrainer,
    "MarigoldNormalsTrainer": MarigoldNormalsTrainer,
    "MarigoldDepthLCMTrainer": MarigoldDepthLCMTrainer

}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]