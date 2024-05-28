# An official reimplemented version of Marigold training script
# Last modified: 2024-05-17
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
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import argparse
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import List

import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from marigold.marigold_pipeline import MarigoldPipeline
from src.dataset import BaseDepthDataset, DatasetMode, get_dataset
from src.dataset.mixed_sampler import MixedBatchSampler
from src.trainer import get_trainer_cls
from src.util.config_util import (
    find_value_in_omegaconf,
    recursive_load_config,
)
from src.util.depth_transform import (
    DepthNormalizerBase,
    get_depth_normalizer,
)
from src.util.logging_util import (
    config_logging,
    init_wandb,
    load_wandb_job_id,
    log_slurm_job_id,
    save_wandb_job_id,
    tb_logger,
)
from src.util.slurm_util import get_local_scratch_dir, is_on_slurm

if "__main__" == __name__:
    t_start = datetime.now()
    print(f"start at {t_start}")

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Train your cute model!")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_marigold.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--resume_run",
        action="store",
        default=None,
        help="Path of checkpoint to be resumed. If given, will ignore --config, and checkpoint in the config",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="directory to save checkpoints"
    )
    parser.add_argument("--no_cuda", action="store_true", help="Do not use cuda.")
    parser.add_argument(
        "--exit_after",
        type=int,
        default=-1,
        help="Save checkpoint and exit after X minutes.",
    )
    parser.add_argument("--no_wandb", action="store_true", help="run without wandb")
    parser.add_argument(
        "--do_not_copy_data",
        action="store_true",
        help="On Slurm cluster, do not copy data to local scratch",
    )
    parser.add_argument(
        "--base_data_dir", type=str, default=None, help="directory of training data"
    )
    parser.add_argument(
        "--base_ckpt_dir",
        type=str,
        default=None,
        help="directory of pretrained checkpoint",
    )
    parser.add_argument(
        "--add_datetime_prefix",
        action="store_true",
        help="Add datetime to the output folder name",
    )

    args = parser.parse_args()
    resume_run = args.resume_run
    output_dir = args.output_dir
    base_data_dir = (
        args.base_data_dir
        if args.base_data_dir is not None
        else os.environ["BASE_DATA_DIR"]
    )
    base_ckpt_dir = (
        args.base_ckpt_dir
        if args.base_ckpt_dir is not None
        else os.environ["BASE_CKPT_DIR"]
    )

    # -------------------- Initialization --------------------
    # Resume previous run
    if resume_run is not None:
        print(f"Resume run: {resume_run}")
        out_dir_run = os.path.dirname(os.path.dirname(resume_run))
        job_name = os.path.basename(out_dir_run)
        # Resume config file
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))
    else:
        # Run from start
        cfg = recursive_load_config(args.config)
        # Full job name
        pure_job_name = os.path.basename(args.config).split(".")[0]
        # Add time prefix
        if args.add_datetime_prefix:
            job_name = f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}-{pure_job_name}"
        else:
            job_name = pure_job_name

        # Output dir
        if output_dir is not None:
            out_dir_run = os.path.join(output_dir, job_name)
        else:
            out_dir_run = os.path.join("./output", job_name)
        os.makedirs(out_dir_run, exist_ok=False)

    cfg_data = cfg.dataset

    # Other directories
    out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")
    if not os.path.exists(out_dir_ckpt):
        os.makedirs(out_dir_ckpt)
    out_dir_tb = os.path.join(out_dir_run, "tensorboard")
    if not os.path.exists(out_dir_tb):
        os.makedirs(out_dir_tb)
    out_dir_eval = os.path.join(out_dir_run, "evaluation")
    if not os.path.exists(out_dir_eval):
        os.makedirs(out_dir_eval)
    out_dir_vis = os.path.join(out_dir_run, "visualization")
    if not os.path.exists(out_dir_vis):
        os.makedirs(out_dir_vis)

    # -------------------- Logging settings --------------------
    config_logging(cfg.logging, out_dir=out_dir_run)
    logging.debug(f"config: {cfg}")

    # Initialize wandb
    if not args.no_wandb:
        if resume_run is not None:
            wandb_id = load_wandb_job_id(out_dir_run)
            wandb_cfg_dic = {
                "id": wandb_id,
                "resume": "must",
                **cfg.wandb,
            }
        else:
            wandb_cfg_dic = {
                "config": dict(cfg),
                "name": job_name,
                "mode": "online",
                **cfg.wandb,
            }
        wandb_cfg_dic.update({"dir": out_dir_run})
        wandb_run = init_wandb(enable=True, **wandb_cfg_dic)
        save_wandb_job_id(wandb_run, out_dir_run)
    else:
        init_wandb(enable=False)

    # Tensorboard (should be initialized after wandb)
    tb_logger.set_dir(out_dir_tb)

    log_slurm_job_id(step=0)

    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"device = {device}")

    # -------------------- Snapshot of code and config --------------------
    if resume_run is None:
        _output_path = os.path.join(out_dir_run, "config.yaml")
        with open(_output_path, "w+") as f:
            OmegaConf.save(config=cfg, f=f)
        logging.info(f"Config saved to {_output_path}")
        # Copy and tar code on the first run
        _temp_code_dir = os.path.join(out_dir_run, "code_tar")
        _code_snapshot_path = os.path.join(out_dir_run, "code_snapshot.tar")
        os.system(
            f"rsync --relative -arhvz --quiet --filter=':- .gitignore' --exclude '.git' . '{_temp_code_dir}'"
        )
        os.system(f"tar -cf {_code_snapshot_path} {_temp_code_dir}")
        os.system(f"rm -rf {_temp_code_dir}")
        logging.info(f"Code snapshot saved to: {_code_snapshot_path}")

    # -------------------- Copy data to local scratch (Slurm) --------------------
    if is_on_slurm() and (not args.do_not_copy_data):
        # local scratch dir
        original_data_dir = base_data_dir
        base_data_dir = os.path.join(get_local_scratch_dir(), "Marigold_data")
        # copy data
        required_data_list = find_value_in_omegaconf("dir", cfg_data)
        # if cfg_train.visualize.init_latent_path is not None:
        #     required_data_list.append(cfg_train.visualize.init_latent_path)
        required_data_list = list(set(required_data_list))
        logging.info(f"Required_data_list: {required_data_list}")
        for d in tqdm(required_data_list, desc="Copy data to local scratch"):
            ori_dir = os.path.join(original_data_dir, d)
            dst_dir = os.path.join(base_data_dir, d)
            os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
            if os.path.isfile(ori_dir):
                shutil.copyfile(ori_dir, dst_dir)
            elif os.path.isdir(ori_dir):
                shutil.copytree(ori_dir, dst_dir)
        logging.info(f"Data copied to: {base_data_dir}")

    # -------------------- Gradient accumulation steps --------------------
    eff_bs = cfg.dataloader.effective_batch_size
    accumulation_steps = eff_bs / cfg.dataloader.max_train_batch_size
    assert int(accumulation_steps) == accumulation_steps
    accumulation_steps = int(accumulation_steps)

    logging.info(
        f"Effective batch size: {eff_bs}, accumulation steps: {accumulation_steps}"
    )

    # -------------------- Data --------------------
    loader_seed = cfg.dataloader.seed
    if loader_seed is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(loader_seed)

    # Training dataset
    depth_transform: DepthNormalizerBase = get_depth_normalizer(
        cfg_normalizer=cfg.depth_normalization
    )
    train_dataset: BaseDepthDataset = get_dataset(
        cfg_data.train,
        base_data_dir=base_data_dir,
        mode=DatasetMode.TRAIN,
        augmentation_args=cfg.augmentation,
        depth_transform=depth_transform,
    )
    logging.debug("Augmentation: ", cfg.augmentation)
    if "mixed" == cfg_data.train.name:
        dataset_ls = train_dataset
        assert len(cfg_data.train.prob_ls) == len(
            dataset_ls
        ), "Lengths don't match: `prob_ls` and `dataset_list`"
        concat_dataset = ConcatDataset(dataset_ls)
        mixed_sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=cfg.dataloader.max_train_batch_size,
            drop_last=True,
            prob=cfg_data.train.prob_ls,
            shuffle=True,
            generator=loader_generator,
        )
        train_loader = DataLoader(
            concat_dataset,
            batch_sampler=mixed_sampler,
            num_workers=cfg.dataloader.num_workers,
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.dataloader.max_train_batch_size,
            num_workers=cfg.dataloader.num_workers,
            shuffle=True,
            generator=loader_generator,
        )
    # Validation dataset
    val_loaders: List[DataLoader] = []
    for _val_dic in cfg_data.val:
        _val_dataset = get_dataset(
            _val_dic,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
        )
        _val_loader = DataLoader(
            dataset=_val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
        )
        val_loaders.append(_val_loader)

    # Visualization dataset
    vis_loaders: List[DataLoader] = []
    for _vis_dic in cfg_data.vis:
        _vis_dataset = get_dataset(
            _vis_dic,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
        )
        _vis_loader = DataLoader(
            dataset=_vis_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
        )
        vis_loaders.append(_vis_loader)

    # -------------------- Model --------------------
    _pipeline_kwargs = cfg.pipeline.kwargs if cfg.pipeline.kwargs is not None else {}
    model = MarigoldPipeline.from_pretrained(
        os.path.join(base_ckpt_dir, cfg.model.pretrained_path), **_pipeline_kwargs
    )

    # -------------------- Trainer --------------------
    # Exit time
    if args.exit_after > 0:
        t_end = t_start + timedelta(minutes=args.exit_after)
        logging.info(f"Will exit at {t_end}")
    else:
        t_end = None

    trainer_cls = get_trainer_cls(cfg.trainer.name)
    logging.debug(f"Trainer: {trainer_cls}")
    trainer = trainer_cls(
        cfg=cfg,
        model=model,
        train_dataloader=train_loader,
        device=device,
        base_ckpt_dir=base_ckpt_dir,
        out_dir_ckpt=out_dir_ckpt,
        out_dir_eval=out_dir_eval,
        out_dir_vis=out_dir_vis,
        accumulation_steps=accumulation_steps,
        val_dataloaders=val_loaders,
        vis_dataloaders=vis_loaders,
    )

    # -------------------- Checkpoint --------------------
    if resume_run is not None:
        trainer.load_checkpoint(
            resume_run, load_trainer_state=True, resume_lr_scheduler=True
        )

    # -------------------- Training & Evaluation Loop --------------------
    try:
        trainer.train(t_end=t_end)
    except Exception as e:
        logging.exception(e)
