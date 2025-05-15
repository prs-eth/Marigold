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

# Adapted from: https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2?s=31

import math
import torch


def multi_res_noise_like(
    x, strength=0.9, downscale_strategy="original", generator=None, device=None
):
    if torch.is_tensor(strength):
        strength = strength.reshape((-1, 1, 1, 1))
    b, c, w, h = x.shape

    if device is None:
        device = x.device

    up_sampler = torch.nn.Upsample(size=(w, h), mode="bilinear")
    noise = torch.randn(x.shape, device=x.device, generator=generator)

    if "original" == downscale_strategy:
        for i in range(10):
            r = (
                torch.rand(1, generator=generator, device=device) * 2 + 2
            )  # Rather than always going 2x,
            w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
            if w == 1 or h == 1:
                break  # Lowest resolution is 1x1
    elif "every_layer" == downscale_strategy:
        for i in range(int(math.log2(min(w, h)))):
            w, h = max(1, int(w / 2)), max(1, int(h / 2))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
    elif "power_of_two" == downscale_strategy:
        for i in range(10):
            r = 2
            w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
            if w == 1 or h == 1:
                break  # Lowest resolution is 1x1
    elif "random_step" == downscale_strategy:
        for i in range(10):
            r = (
                torch.rand(1, generator=generator, device=device) * 2 + 2
            )  # Rather than always going 2x,
            w, h = max(1, int(w / (r))), max(1, int(h / (r)))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
            if w == 1 or h == 1:
                break  # Lowest resolution is 1x1
    else:
        raise ValueError(f"unknown downscale strategy: {downscale_strategy}")

    noise = noise / noise.std()  # Scaled back to roughly unit variance
    return noise
