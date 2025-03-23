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

import logging
import os
import sys
import wandb
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter


def config_logging(cfg_logging, out_dir=None):
    file_level = cfg_logging.get("file_level", 10)
    console_level = cfg_logging.get("console_level", 10)

    log_formatter = logging.Formatter(cfg_logging["format"])

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    root_logger.setLevel(min(file_level, console_level))

    if out_dir is not None:
        _logging_file = os.path.join(
            out_dir, cfg_logging.get("filename", "logging.log")
        )
        file_handler = logging.FileHandler(_logging_file)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_level)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)

    # Avoid pollution by packages
    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)


class MyTrainingLogger:
    """Tensorboard + wandb logger"""

    writer: SummaryWriter
    is_initialized = False

    def __init__(self) -> None:
        pass

    def set_dir(self, tb_log_dir):
        if self.is_initialized:
            raise ValueError("Do not initialize writer twice")
        self.writer = SummaryWriter(tb_log_dir)
        self.is_initialized = True

    def log_dict(self, scalar_dict, global_step, walltime=None):
        for k, v in scalar_dict.items():
            self.writer.add_scalar(k, v, global_step=global_step, walltime=walltime)
        return


# global instance
tb_logger = MyTrainingLogger()


# -------------- wandb tools --------------
def init_wandb(enable: bool, **kwargs):
    if enable:
        run = wandb.init(sync_tensorboard=True, **kwargs)
    else:
        run = wandb.init(mode="disabled")
    return run


def log_slurm_job_id(step):
    global tb_logger
    _jobid = os.getenv("SLURM_JOB_ID")
    if _jobid is None:
        _jobid = -1
    tb_logger.writer.add_scalar("job_id", int(_jobid), global_step=step)
    logging.debug(f"Slurm job_id: {_jobid}")


def load_wandb_job_id(out_dir):
    with open(os.path.join(out_dir, "WANDB_ID"), "r") as f:
        wandb_id = f.read()
    return wandb_id


def save_wandb_job_id(run, out_dir):
    with open(os.path.join(out_dir, "WANDB_ID"), "w+") as f:
        f.write(run.id)


def eval_dict_to_text(val_metrics: dict, dataset_name: str, sample_list_path: str):
    eval_text = f"Evaluation metrics:\n\
     on dataset: {dataset_name}\n\
     over samples in: {sample_list_path}\n"

    eval_text += tabulate([val_metrics.keys(), val_metrics.values()])
    return eval_text
