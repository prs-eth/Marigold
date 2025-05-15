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
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
from tqdm.auto import tqdm

from marigold.util.image_util import srgb2linear, linear2srgb
from src.dataset import DatasetMode, get_dataset
from src.util.metric import MetricTracker, compute_iid_metric


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Marigold : Intrinsic Image Decomposition : Metrics Evaluation"
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        required=True,
        help="Directory with predictions obtained from inference.",
    )
    parser.add_argument(
        "--target_names",
        nargs="+",
        default=["albedo", "material"],
        type=str,
        help="A list of predicted targets to evaluate.",
    )
    parser.add_argument(
        "--targets_to_eval_in_linear_space",
        nargs="*",
        default=[None],
        type=str,
        help="A list of targets to evaluate in linear space (as opposed to sRGB by default). Defaults to an empty list.",
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

    # -------------------- Initialize Metrics --------------------
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    target_names = args.target_names

    targets_to_eval_in_linear_space = args.targets_to_eval_in_linear_space
    for tn in targets_to_eval_in_linear_space:
        if tn is not None and tn not in target_names:
            raise ValueError(
                f"'{tn}' specified in targets_to_eval_in_linear_space does not belong to the predicted targets: "
                f"{target_names=}"
            )

    metrics_dict = {}
    for target_name in target_names:
        metrics_dict["psnr_" + target_name] = psnr_metric
        metrics_dict["ssim_" + target_name] = ssim_metric
        metrics_dict["lpips_" + target_name] = lpips_metric

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(args.dataset_config)

    dataset = get_dataset(
        cfg_data, base_data_dir=args.base_data_dir, mode=DatasetMode.EVAL
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # -------------------- Eval metrics --------------------
    metric_tracker = MetricTracker(*metrics_dict.keys())
    metric_tracker.reset()

    # -------------------- Results Dictionary --------------------
    results = {}

    # -------------------- Per-sample metrics file --------------------
    per_sample_filename = os.path.join(args.output_dir, "per_sample_metrics.csv")
    # write title
    with open(per_sample_filename, "w+") as f:
        f.write("filename,")
        f.write(",".join(metrics_dict.keys()))
        f.write("\n")

    # -------------------- Evaluate --------------------
    for data in tqdm(dataloader, desc="Evaluating"):
        rgb_name = data["rgb_relative_path"][0]

        # load predictions
        rgb_basename = os.path.basename(rgb_name)
        scene_dir = os.path.join(args.prediction_dir, os.path.dirname(rgb_name))
        rgb_basename_without_extension = os.path.splitext(rgb_basename)[0]

        sample_metric = []

        for target_name in target_names:
            target_gt = data[target_name].to(device)

            pred_basename_target = (
                rgb_basename_without_extension + "_" + target_name + ".npy"
            )
            pred_path_target = os.path.join(scene_dir, pred_basename_target)
            if not os.path.exists(pred_path_target):
                logging.warning(f"Can't find prediction: {pred_path_target}")
                continue
            target_pred = (
                torch.from_numpy(np.load((pred_path_target))).unsqueeze(0).to(device)
            )  # [1,3,H,W]

            # IID Appearance model predicts all modalities in sRGB space
            if target_name in targets_to_eval_in_linear_space:
                target_gt = srgb2linear(target_gt)
                target_pred = srgb2linear(target_pred)

            # Hypersim GT and IID Lighting model predictions are in linear space
            # We evaluate albedo in sRGB space
            if (
                "hypersim" in cfg_data.name
                and len(target_names) == 3
                and target_name == "albedo"
            ):
                # linear --> sRGB
                target_gt = linear2srgb(target_gt)
                target_pred = linear2srgb(target_pred)

            for metric_name in ("psnr", "ssim", "lpips"):
                _metric_name = metric_name + "_" + target_name

                if args.use_mask:
                    _mask_name = "mask" + "_" + target_name
                    valid_mask = data[_mask_name].to(device)
                else:
                    valid_mask = None
                _metric = compute_iid_metric(
                    target_pred,
                    target_gt,
                    target_name,
                    metric_name,
                    metrics_dict[_metric_name],
                    valid_mask,
                )
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
