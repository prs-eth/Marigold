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

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
import numpy as np
import os
import torch
from omegaconf import OmegaConf
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import DatasetMode, get_dataset
from src.util import metric
from src.util.metric import MetricTracker, compute_cosine_error

eval_metrics = [
    "mean_angular_error",
    "median_angular_error",
    "sub5_error",
    "sub7_5_error",
    "sub11_25_error",
    "sub22_5_error",
    "sub30_error",
]

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Marigold : Surface Normals Estimation : Metrics Evaluation"
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        required=True,
        help="Directory with predictions obtained from inference.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to the config file of the evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Base path to the datasets.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--use_mask", action="store_true", help="Evaluate only in the masked region."
    )
    parser.add_argument("--no_cuda", action="store_true", help="Run without cuda.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"Device: {device}")

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(args.dataset_config)

    dataset = get_dataset(
        cfg_data, base_data_dir=args.base_data_dir, mode=DatasetMode.EVAL
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # -------------------- Eval metrics --------------------
    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

    metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker.reset()

    # -------------------- Results Dictionary --------------------
    results = {}

    # -------------------- Per-sample metrics file --------------------
    per_sample_filename = os.path.join(args.output_dir, "per_sample_metrics.csv")
    # write title
    with open(per_sample_filename, "w+") as f:
        f.write("filename,")
        f.write(",".join([m.__name__ for m in metric_funcs]))
        f.write("\n")

    # -------------------- Evaluate --------------------
    for data in tqdm(dataloader, desc="Evaluating"):
        # GT data
        rgb_name = data["rgb_relative_path"][0]
        normals_gt = data["normals"].to(device)  # [1,3,H,W]

        # Load predictions
        rgb_basename = os.path.basename(rgb_name)
        scene_dir = os.path.join(args.prediction_dir, os.path.dirname(rgb_name))
        rgb_basename_without_extension = os.path.splitext(rgb_basename)[0]
        pred_basename = rgb_basename_without_extension + ".npy"
        pred_path = os.path.join(scene_dir, pred_basename)

        if not os.path.exists(pred_path):
            logging.warning(f"Can't find prediction: {pred_path}")
            continue

        normals_pred = (
            torch.from_numpy(np.load((pred_path)).astype(np.float32))
            .unsqueeze(0)
            .to(device)
        )  # [1,3,H,W]
        cosine_error = compute_cosine_error(normals_pred, normals_gt, masked=True)
        sample_metric = []

        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(cosine_error).item()
            sample_metric.append(_metric.__str__())
            metric_tracker.update(_metric_name, _metric)

        # Save per-sample metric
        with open(per_sample_filename, "a+") as f:
            f.write(rgb_name + ",")
            f.write(",".join(sample_metric))
            f.write("\n")

    # -------------------- Save metrics to file --------------------
    eval_text = f"Evaluation metrics:\n\
    of predictions: {args.prediction_dir}\n\
    on dataset: {dataset.disp_name}\n\
    with samples in: {dataset.filename_ls_path}\n"

    eval_text += tabulate(
        [metric_tracker.result().keys(), metric_tracker.result().values()]
    )

    metrics_filename = "eval_metrics"
    metrics_filename += ".txt"

    _save_to = os.path.join(args.output_dir, metrics_filename)
    with open(_save_to, "w+") as f:
        f.write(eval_text)
        logging.info(f"Evaluation metrics saved to {_save_to}")
