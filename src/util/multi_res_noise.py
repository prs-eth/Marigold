# Author: Bingxin Ke
# Last modified: 2024-04-18

import torch
import math


# adapted from: https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2?s=31
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
