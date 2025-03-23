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

import argparse
import math
import h5py
import numpy as np
import os
import cv2
import pandas as pd
import functools
import json
from pathlib import Path
from tqdm import tqdm

from hypersim_util import tone_map_hypersim


def psnr(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    mse = np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def tone_map(img, scale):
    gamma = 1.0 / 2.2
    img_scaled = scale * img
    img_scaled_clipped = np.maximum(img_scaled, 0)
    mask_scaled_negative = img_scaled != img_scaled_clipped
    assert (
        np.count_nonzero(mask_scaled_negative) == 0
    ), f"{img_scaled[mask_scaled_negative]=}"
    img = np.power(img_scaled_clipped, gamma)
    img = np.clip(img, 0, 1)
    return img


def compute_tone_map_scale(img, valid_mask=None):
    inv_gamma = 2.2
    percentile = 90
    brightness_nth_percentile_desired = 0.8

    brightness = 0.3 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 2]
    brightness = brightness

    if valid_mask is not None:
        num_valid = np.count_nonzero(valid_mask)
        if num_valid == 0:
            return 1.0
        brightness = brightness[valid_mask]
    else:
        brightness = brightness.flatten()

    eps = 0.0001
    brightness_nth_percentile_current = np.percentile(brightness, percentile)

    if brightness_nth_percentile_current < eps:
        return 0.0

    scale = float(
        np.power(brightness_nth_percentile_desired, inv_gamma)
        / brightness_nth_percentile_current
    )

    return scale


def get_path_stem(row, dataset_dir, modality):
    return os.path.join(
        dataset_dir,
        row.scene_name,
        "images",
        f"scene_{row.camera_name}_{modality}_hdf5",
        f"frame.{row.frame_id:04d}",
    )


def touch(path, msg=None):
    if msg is None:
        Path(path).touch()
    else:
        with open(path, "w") as f:
            f.write(msg)


def finalize(path_marker, tmp_dir, out_dir, msg=None):
    touch(path_marker, msg=msg)
    return None


def process_sample(row, dataset_dir, target_dir, split):
    path_stem_final = get_path_stem(row, dataset_dir, "final")
    path_stem_geometry = get_path_stem(row, dataset_dir, "geometry")

    out_dir = os.path.join(target_dir, row.scene_name)
    tmp_dir = out_dir + "_invalid"
    status_file = os.path.join(
        tmp_dir, f"status_{row.camera_name}_fr{row.frame_id:04d}.txt"
    )

    os.makedirs(tmp_dir, exist_ok=True)

    path_color = path_stem_final + ".color.hdf5"
    path_albedo = path_stem_final + ".diffuse_reflectance.hdf5"
    path_shading = path_stem_final + ".diffuse_illumination.hdf5"
    path_residual = path_stem_final + ".residual.hdf5"
    path_render_entity_id = path_stem_geometry + ".render_entity_id.hdf5"

    with h5py.File(path_render_entity_id, "r") as f:
        render_entity_id = np.array(f["dataset"]).astype(int)

    mask_invalid = render_entity_id == -1
    num_invalid = np.count_nonzero(mask_invalid)
    if num_invalid > 0:
        return finalize(status_file, tmp_dir, out_dir, f"{num_invalid=}")

    with h5py.File(path_color, "r") as f:
        color = np.array(f["dataset"]).astype(float)
    with h5py.File(path_albedo, "r") as f:
        albedo = np.array(f["dataset"]).astype(float)
    with h5py.File(path_shading, "r") as f:
        shading = np.array(f["dataset"]).astype(float)
    with h5py.File(path_residual, "r") as f:
        residual = np.array(f["dataset"]).astype(float)

    stats = {
        "albedo_min": np.min(albedo),
        "albedo_max": np.max(albedo),
        "albedo_mean": np.mean(albedo),
        "albedo_std": np.std(albedo),
        "albedo_98": np.percentile(albedo, 98),
        "shading_min": np.min(shading),
        "shading_max": np.max(shading),
        "shading_mean": np.mean(shading),
        "shading_std": np.std(shading),
        "shading_98": np.percentile(shading, 98),
        "residual_min": np.min(residual),
        "residual_max": np.max(residual),
        "residual_mean": np.mean(residual),
        "residual_std": np.std(residual),
        "residual_02": np.percentile(residual, 2),
        "residual_98": np.percentile(residual, 98),
    }

    # do checks and reject non-valid samples in the train and val split
    if split != "test":
        shading_computed_color = albedo * shading + residual
        if not np.isfinite(shading_computed_color).all():
            return finalize(
                status_file,
                tmp_dir,
                out_dir,
                "shading_computed_color has non-finite values",
            )

        nan_stats = {
            "albedo_nan": np.isnan(albedo),
            "shading_nan": np.isnan(shading),
            "residual_nan": np.isnan(residual),
        }

        if np.any(nan_stats["albedo_nan"]):
            return finalize(
                status_file, tmp_dir, out_dir, f"{nan_stats['albedo_nan'].sum()}"
            )

        if np.any(nan_stats["shading_nan"]):
            return finalize(
                status_file, tmp_dir, out_dir, f"{nan_stats['shading_nan'].sum()}"
            )

        if np.any(nan_stats["residual_nan"]):
            return finalize(
                status_file, tmp_dir, out_dir, f"{nan_stats['residual_nan'].sum()}"
            )

        if stats["albedo_min"] < 0 or stats["albedo_max"] > 1:
            return finalize(
                status_file,
                tmp_dir,
                out_dir,
                f"{stats['albedo_min']:0.4f} {stats['albedo_max']:0.4f}",
            )
        if stats["shading_min"] < 0:
            return finalize(
                status_file, tmp_dir, out_dir, f"{stats['shading_min']:0.4f}"
            )
        if stats["residual_min"] < 0:
            return finalize(
                status_file, tmp_dir, out_dir, f"{stats['residual_min']:0.4f}"
            )

        color_tmscale = compute_tone_map_scale(color)
        if not math.isfinite(color_tmscale):
            return finalize(
                status_file,
                tmp_dir,
                out_dir,
                f"Tone mapping scale is not finite: {color_tmscale}",
            )
        color_tm = tone_map(color, color_tmscale)
        color_tm = (color_tm * 255).astype(np.uint8)

        residual_cutoff = stats[
            "shading_98"
        ]  # not a mistake, we want shading and residual to be on the same scale

        clipped_shading = np.clip(shading, 0, stats["shading_98"])
        clipped_residual = np.clip(residual, 0, residual_cutoff)
        clipped_computed_color = albedo * clipped_shading + clipped_residual
        clipped_computed_color_tmscale = compute_tone_map_scale(clipped_computed_color)
        clipped_computed_color_tm = tone_map(
            clipped_computed_color, clipped_computed_color_tmscale
        )
        clipped_computed_color_tm = (clipped_computed_color_tm * 255).astype(np.uint8)
        clipped_color_tm_psnr = psnr(color_tm, clipped_computed_color_tm)

        if clipped_color_tm_psnr < 40:
            return finalize(
                status_file, tmp_dir, out_dir, f"{clipped_color_tm_psnr=:0.4f}"
            )

    os.makedirs(out_dir, exist_ok=True)

    # Tone map
    rgb_color_tm = tone_map_hypersim(color, render_entity_id)
    rgb_int = (rgb_color_tm * 255).astype(np.uint8)  # [H, W, RGB]

    # Save RGB
    rgb_name = f"rgb_{row.camera_name}_fr{row.frame_id:04d}.png"
    rgb_path = os.path.join(out_dir, rgb_name)
    cv2.imwrite(
        rgb_path,
        cv2.cvtColor(rgb_int, cv2.COLOR_RGB2BGR),
    )
    # save albedo
    albedo_name = f"albedo_{row.camera_name}_fr{row.frame_id:04d}.npy"
    albedo_path = os.path.join(out_dir, albedo_name)
    np.save(albedo_path, np.clip(albedo, 0, 1.0))

    # save shading
    shading_name = f"shading_{row.camera_name}_fr{row.frame_id:04d}.npy"
    shading_path = os.path.join(out_dir, shading_name)
    np.save(shading_path, shading)

    residual_name = f"residual_{row.camera_name}_fr{row.frame_id:04d}.npy"
    residual_path = os.path.join(out_dir, residual_name)
    np.save(residual_path, residual)

    shading_stats_name = f"shading_stats_{row.camera_name}_fr{row.frame_id:04d}.json"
    shading_stats_path = os.path.join(out_dir, shading_stats_name)
    with open(shading_stats_path, "w") as fp:
        json.dump(stats, fp)

    out = {
        "path_rgb": os.path.join("split_placeholder", row.scene_name, rgb_name),
        "path_albedo": os.path.join("split_placeholder", row.scene_name, albedo_name),
        "path_shading": os.path.join("split_placeholder", row.scene_name, shading_name),
        "path_residual": os.path.join(
            "split_placeholder", row.scene_name, residual_name
        ),
        "path_stats": os.path.join(
            "split_placeholder", row.scene_name, shading_stats_name
        ),
    }

    return out


def process_split(split, df, dataset_dir, output_dir):
    target_dir = os.path.join(output_dir, split)
    partial_process_sample = functools.partial(
        process_sample, dataset_dir=dataset_dir, target_dir=target_dir, split=split
    )
    df = df[df.split_partition_name == split]
    out = []

    for _, df_row in tqdm(df.iterrows(), total=len(df)):
        result = partial_process_sample(df_row)
        if result is not None:
            assert isinstance(result, dict)
            out.append(result)

    with open(os.path.join(output_dir, f"filename_list_{split}.txt"), "w") as f:
        lines = [
            f"{o['path_rgb'].replace('split_placeholder', split)} {o['path_albedo'].replace('split_placeholder', split)} {o['path_shading'].replace('split_placeholder', split)} {o['path_residual'].replace('split_placeholder', split)} {o['path_stats'].replace('split_placeholder', split)}"
            for o in out
        ]
        f.writelines("\n".join(lines))


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_csv",
        type=str,
        default="data/hypersim/metadata_images_split_scene_v1.csv",
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    split_csv = args.split_csv
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    df = pd.read_csv(split_csv)
    df = df[df.included_in_public_release]
    for split in ["train", "val", "test"]:
        process_split(split, df, dataset_dir, output_dir)
    print("Preprocessing finished")
