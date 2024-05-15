# Author: Bingxin Ke
# Last modified: 2024-02-19

import argparse
import os

import cv2
import h5py
import numpy as np
import pandas as pd
from hypersim_util import dist_2_depth, tone_map
from tqdm import tqdm

IMG_WIDTH = 1024
IMG_HEIGHT = 768
FOCAL_LENGTH = 886.81

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_csv",
        type=str,
        default="data/Hypersim/metadata_images_split_scene_v1.csv",
    )
    parser.add_argument("--dataset_dir", type=str, default="data/Hypersim/raw_data")
    parser.add_argument("--output_dir", type=str, default="data/Hypersim/processed")

    args = parser.parse_args()

    split_csv = args.split_csv
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # %%
    raw_meta_df = pd.read_csv(split_csv)
    meta_df = raw_meta_df[raw_meta_df.included_in_public_release].copy()

    # %%
    for split in ["train", "val", "test"]:
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir)

        split_meta_df = meta_df[meta_df.split_partition_name == split].copy()
        split_meta_df["rgb_path"] = None
        split_meta_df["rgb_mean"] = np.nan
        split_meta_df["rgb_std"] = np.nan
        split_meta_df["rgb_min"] = np.nan
        split_meta_df["rgb_max"] = np.nan
        split_meta_df["depth_path"] = None
        split_meta_df["depth_mean"] = np.nan
        split_meta_df["depth_std"] = np.nan
        split_meta_df["depth_min"] = np.nan
        split_meta_df["depth_max"] = np.nan
        split_meta_df["invalid_ratio"] = np.nan

        for i, row in tqdm(split_meta_df.iterrows(), total=len(split_meta_df)):
            # Load data
            rgb_path = os.path.join(
                row.scene_name,
                "images",
                f"scene_{row.camera_name}_final_hdf5",
                f"frame.{row.frame_id:04d}.color.hdf5",
            )
            dist_path = os.path.join(
                row.scene_name,
                "images",
                f"scene_{row.camera_name}_geometry_hdf5",
                f"frame.{row.frame_id:04d}.depth_meters.hdf5",
            )
            render_entity_id_path = os.path.join(
                row.scene_name,
                "images",
                f"scene_{row.camera_name}_geometry_hdf5",
                f"frame.{row.frame_id:04d}.render_entity_id.hdf5",
            )
            assert os.path.exists(os.path.join(dataset_dir, rgb_path))
            assert os.path.exists(os.path.join(dataset_dir, dist_path))

            with h5py.File(os.path.join(dataset_dir, rgb_path), "r") as f:
                rgb = np.array(f["dataset"]).astype(float)
            with h5py.File(os.path.join(dataset_dir, dist_path), "r") as f:
                dist_from_center = np.array(f["dataset"]).astype(float)
            with h5py.File(os.path.join(dataset_dir, render_entity_id_path), "r") as f:
                render_entity_id = np.array(f["dataset"]).astype(int)

            # Tone map
            rgb_color_tm = tone_map(rgb, render_entity_id)
            rgb_int = (rgb_color_tm * 255).astype(np.uint8)  # [H, W, RGB]

            # Distance -> depth
            plane_depth = dist_2_depth(
                IMG_WIDTH, IMG_HEIGHT, FOCAL_LENGTH, dist_from_center
            )
            valid_mask = render_entity_id != -1

            # Record invalid ratio
            invalid_ratio = (np.prod(valid_mask.shape) - valid_mask.sum()) / np.prod(
                valid_mask.shape
            )
            plane_depth[~valid_mask] = 0

            # Save as png
            scene_path = row.scene_name
            if not os.path.exists(os.path.join(split_output_dir, row.scene_name)):
                os.makedirs(os.path.join(split_output_dir, row.scene_name))

            rgb_name = f"rgb_{row.camera_name}_fr{row.frame_id:04d}.png"
            rgb_path = os.path.join(scene_path, rgb_name)
            cv2.imwrite(
                os.path.join(split_output_dir, rgb_path),
                cv2.cvtColor(rgb_int, cv2.COLOR_RGB2BGR),
            )

            plane_depth *= 1000.0
            plane_depth = plane_depth.astype(np.uint16)
            depth_name = f"depth_plane_{row.camera_name}_fr{row.frame_id:04d}.png"
            depth_path = os.path.join(scene_path, depth_name)
            cv2.imwrite(os.path.join(split_output_dir, depth_path), plane_depth)

            # Meta data
            split_meta_df.at[i, "rgb_path"] = rgb_path
            split_meta_df.at[i, "rgb_mean"] = np.mean(rgb_int)
            split_meta_df.at[i, "rgb_std"] = np.std(rgb_int)
            split_meta_df.at[i, "rgb_min"] = np.min(rgb_int)
            split_meta_df.at[i, "rgb_max"] = np.max(rgb_int)

            split_meta_df.at[i, "depth_path"] = depth_path
            restored_depth = plane_depth / 1000.0
            split_meta_df.at[i, "depth_mean"] = np.mean(restored_depth)
            split_meta_df.at[i, "depth_std"] = np.std(restored_depth)
            split_meta_df.at[i, "depth_min"] = np.min(restored_depth)
            split_meta_df.at[i, "depth_max"] = np.max(restored_depth)

            split_meta_df.at[i, "invalid_ratio"] = invalid_ratio

        with open(
            os.path.join(split_output_dir, f"filename_list_{split}.txt"), "w+"
        ) as f:
            lines = split_meta_df.apply(
                lambda r: f"{r['rgb_path']} {r['depth_path']}", axis=1
            ).tolist()
            f.writelines("\n".join(lines))

        with open(
            os.path.join(split_output_dir, f"filename_meta_{split}.csv"), "w+"
        ) as f:
            split_meta_df.to_csv(f, header=True)

    print("Preprocess finished")
