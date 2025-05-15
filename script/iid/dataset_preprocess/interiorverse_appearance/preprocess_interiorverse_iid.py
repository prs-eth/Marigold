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
import os
import tarfile
from tqdm import tqdm


def create_interiorverse_tar(raw_data_dir, base_data_dir):
    # File types we want to include
    target_types = ["im", "mask", "albedo", "material"]

    # Setup paths
    scenes_path = os.path.join(raw_data_dir, "scenes_85")
    output_dir = os.path.join(base_data_dir, "interiorverse")
    os.makedirs(output_dir, exist_ok=True)
    output_tar = os.path.join(output_dir, "InteriorVerse.tar")

    # Create tar file
    with tarfile.open(output_tar, "w") as tar:
        # Get scene directories in scenes_85
        scene_dirs = os.listdir(scenes_path)

        # Process each scene directory with progress bar
        for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
            scene_path = os.path.join(scenes_path, scene_dir)

            # Process files in scene directory
            if os.path.isdir(scene_path):
                files = [
                    f
                    for f in os.listdir(scene_path)
                    if any(f"_{type}." in f for type in target_types)
                ]

                # Add files to tar
                for file in files:
                    full_path = os.path.join(scene_path, file)
                    arcname = os.path.join("scenes_85", scene_dir, file)
                    tar.add(full_path, arcname=arcname)
                    print(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create tar file for InteriorVerse dataset"
    )
    parser.add_argument(
        "--output_data_dir",
        type=str,
        required=True,
        help="Path to base data directory. The same as in train.py, infer.py or eval.py",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        required=True,
        help="The path of raw InteriorVerse dataset. The same as your/raw/InteriorVerse/ in README.md",
    )
    args = parser.parse_args()

    print(
        f"Creating tar file under: {os.path.join(args.output_data_dir, 'interiorverse')}"
    )
    create_interiorverse_tar(args.raw_data_dir, args.output_data_dir)
    print("Done!")
