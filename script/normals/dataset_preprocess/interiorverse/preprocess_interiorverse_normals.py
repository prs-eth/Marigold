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
import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # we only use scenes_85
    scenes85_input_dir = os.path.join(dataset_dir, "scenes_85")
    scenes85_output_dir = os.path.join(output_dir, "scenes_85")

    if not os.path.exists(scenes85_output_dir):
        os.makedirs(scenes85_output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "interiorverse_filtered_all.txt"), "w+") as f:
        for scene in tqdm(os.listdir(scenes85_input_dir)):
            for file in os.listdir(os.path.join(scenes85_input_dir, scene)):
                # skip if nor RGB or normals
                if "im.exr" not in file and "normal.exr" not in file:
                    continue

                img_path = os.path.join(scenes85_input_dir, scene, file)
                im = cv2.imread(
                    img_path, -1
                )  # im will be an numpy.float32 array of shape (H, W, 3)
                im = cv2.cvtColor(
                    im, cv2.COLOR_BGR2RGB
                )  # cv2 reads image in BGR shape, convert into RGB
                # skip if image/normal map contains nan values
                if np.any(np.isnan(im)):
                    continue

                # if RGB image
                if "im" in file:
                    im = im.clip(0, 1) ** (
                        1 / 2.2
                    )  # Convert from HDR to LDR with clipping and gamma correction
                    img = (im * 255).astype(np.uint8)
                    img = Image.fromarray(img)

                    rgb_name = file.replace("im.exr", "img.png")
                    if not os.path.exists(os.path.join(scenes85_output_dir, scene)):
                        os.makedirs(
                            os.path.join(scenes85_output_dir, scene), exist_ok=True
                        )
                    rgb_path = os.path.join(scenes85_output_dir, scene, rgb_name)
                    img.save(rgb_path)

                elif "normal" in file:
                    # invalid pixels are 0
                    # skip if normal map contains invalid values
                    invalid_mask = np.linalg.norm(im, axis=2) < 0.1
                    if invalid_mask.sum() > 0:
                        continue

                    # normalize to unit length
                    im /= np.linalg.norm(im, axis=2, keepdims=True)

                    # save as .npy
                    normal_name = file.replace("normal.exr", "normal.npy")
                    if not os.path.exists(os.path.join(scenes85_output_dir, scene)):
                        os.makedirs(
                            os.path.join(scenes85_output_dir, scene), exist_ok=True
                        )
                    normal_path = os.path.join(scenes85_output_dir, scene, normal_name)
                    np.save(normal_path, im)
                    rgb_name = file.replace("normal.exr", "img.png")

                    f.write(
                        f"{os.path.join(scene, rgb_name)} {os.path.join(scene, normal_name)}\n"
                    )

    print("Preprocess finished")
