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

from pylab import count_nonzero, clip, np


# Adapted from https://github.com/apple/ml-hypersim/blob/main/code/python/tools/scene_generate_images_tonemap.py
def tone_map(rgb, entity_id_map):
    assert (entity_id_map != 0).all()

    gamma = 1.0 / 2.2  # standard gamma correction exponent
    inv_gamma = 1.0 / gamma
    percentile = (
        90  # we want this percentile brightness value in the unmodified image...
    )
    brightness_nth_percentile_desired = 0.8  # ...to be this bright after scaling

    valid_mask = entity_id_map != -1

    if count_nonzero(valid_mask) == 0:
        scale = 1.0  # if there are no valid pixels, then set scale to 1.0
    else:
        brightness = (
            0.3 * rgb[:, :, 0] + 0.59 * rgb[:, :, 1] + 0.11 * rgb[:, :, 2]
        )  # "CCIR601 YIQ" method for computing brightness
        brightness_valid = brightness[valid_mask]

        eps = 0.0001  # if the kth percentile brightness value in the unmodified image is less than this, set the scale to 0.0 to avoid divide-by-zero
        brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:
            # Snavely uses the following expression in the code at https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
            # scale = np.exp(np.log(brightness_nth_percentile_desired)*inv_gamma - np.log(brightness_nth_percentile_current))
            #
            # Our expression below is equivalent, but is more intuitive, because it follows more directly from the expression:
            # (scale*brightness_nth_percentile_current)^gamma = brightness_nth_percentile_desired

            scale = (
                np.power(brightness_nth_percentile_desired, inv_gamma)
                / brightness_nth_percentile_current
            )

    rgb_color_tm = np.power(np.maximum(scale * rgb, 0), gamma)
    rgb_color_tm = clip(rgb_color_tm, 0, 1)
    return rgb_color_tm


# According to https://github.com/apple/ml-hypersim/issues/9
def dist_2_depth(width, height, flt_focal, distance):
    img_plane_x = (
        np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width)
        .reshape(1, width)
        .repeat(height, 0)
        .astype(np.float32)[:, :, None]
    )
    img_plane_y = (
        np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height)
        .reshape(height, 1)
        .repeat(width, 1)
        .astype(np.float32)[:, :, None]
    )
    img_plane_z = np.full([height, width, 1], flt_focal, np.float32)
    img_plane = np.concatenate([img_plane_x, img_plane_y, img_plane_z], 2)

    depth = distance / np.linalg.norm(img_plane, 2, 2) * flt_focal
    return depth
