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

import logging
import numpy as np
import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_iid
from .util.image_util import (
    chw2hwc,
    get_tv_resample_method,
    resize_max_res,
)


@dataclass
class IIDEntry:
    """
    A single entry in the IID output, representing one decomposed component.
    For each entry we output the following properties:
        name (`str`):
            The name of the entry.
        array (`np.ndarray`):
            Predicted numpy array with the shape of [3, H, W] values in the range of [0, 1].
        image (`PIL.Image.Image`):
            Predicted image with the shape of [H, W, 3] and values in [0, 255].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty from ensembling.
    """

    name: str
    array: Optional[np.ndarray] = None
    image: Optional[Image.Image] = None
    uncertainty: Optional[np.ndarray] = None


class MarigoldIIDOutput:
    """Output class for Marigold Intrinsic Image Decomposition pipelines."""

    def __init__(self, target_names: List[str]):
        """Initialize output container with target names.

        Args:
            target_names: List of names for each target component
        """
        self.n_targets = len(target_names)
        self.target_names = target_names
        self.entries: List[IIDEntry] = [IIDEntry(name=name) for name in target_names]
        self._entry_map = {entry.name: entry for entry in self.entries}
        self._filled_entries = set()

    def fill_entry(
        self,
        name: str,
        prediction: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        target_properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fill a single entry with prediction data.

        Args:
            name: Name of the entry to fill
            prediction: Tensor containing the prediction for this entry
            uncertainty: Optional tensor containing uncertainty values
            target_properties: Properties of the predicted targets
        """
        if name not in self._entry_map:
            raise KeyError(f"Unknown entry name: {name}")
        if name in self._filled_entries:
            raise RuntimeError(f"Entry {name} already filled")

        entry = self._entry_map[name]

        # Process prediction
        array = prediction.squeeze().cpu().numpy()
        img_array = array

        # Prepare image visualization
        prediction_space = target_properties[name].get("prediction_space", "srgb")
        if prediction_space == "stack":
            pass
        elif prediction_space == "linear":
            up_to_scale = target_properties[name].get("up_to_scale", False)
            if up_to_scale:
                img_array = img_array / max(img_array.max(), 1e-6)
            img_array = img_array ** (1 / 2.2)
        elif prediction_space == "srgb":
            pass

        # Create image
        img_array = (img_array * 255).astype(np.uint8)
        img_array = chw2hwc(img_array)  # Convert from CHW to HWC format
        image = Image.fromarray(img_array)

        # Process uncertainty if available
        uncert_array = (
            uncertainty.squeeze().cpu().numpy() if uncertainty is not None else None
        )

        # Update entry
        entry.array = array
        entry.image = image
        entry.uncertainty = uncert_array

        self._filled_entries.add(name)

    @property
    def is_complete(self) -> bool:
        """Check if all entries have been filled."""
        return len(self._filled_entries) == self.n_targets

    def __getitem__(self, key: str) -> IIDEntry:
        """Get an entry by name."""
        return self._entry_map[key]

    def __iter__(self):
        """Iterate over entries."""
        return iter(self.entries)


class MarigoldIIDPipeline(DiffusionPipeline):
    """
    Pipeline for Marigold Intrinsic Image Decomposition (IID): https://marigoldcomputervision.github.io.
    This class supports arbitrary number of target modalities with names set in `target_names`.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the prediction latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and predictions
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        target_properties (`Dict[str, Any]`, *optional*):
            Properties of the predicted modalities, such as `target_names`, a `List[str]` used to define the number,
            order and names of the predicted modalities, and any other metadata that may be required to interpret the
            predictions.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        target_properties: Optional[Dict[str, Any]] = None,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.register_to_config(
            target_properties=target_properties,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.target_properties = target_properties
        self.target_names = target_properties["target_names"]
        self.n_targets = len(self.target_names)

        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
    ) -> MarigoldIIDOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection.
            ensemble_size (`int`, *optional*, defaults to `1`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize the prediction to match the input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and predictions. This can be one of `bilinear`, `bicubic` or
                `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldIIDOutput`: Output class for Marigold Intrinsic Image Decomposition prediction pipeline.
        """
        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution
        assert processing_res >= 0
        assert ensemble_size >= 1

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting IID -----------------
        # Batch repeated input image
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict IID maps (batched)
        target_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            target_pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
            )
            assert (
                target_pred_raw.dim() == 4
                and target_pred_raw.shape[1] == 3 * self.n_targets
            )
            target_pred_ls.append(target_pred_raw.detach())
        target_preds = torch.concat(target_pred_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            final_pred, pred_uncert = ensemble_iid(
                target_preds,
                **(ensemble_kwargs or {}),
            )
        else:
            final_pred = target_preds
            pred_uncert = None

        # Resize back to original resolution
        if match_input_res:
            final_pred = resize(
                final_pred,
                input_size[-2:],
                interpolation=resample_method,
                antialias=True,
            )

        # Create output
        output = MarigoldIIDOutput(target_names=self.target_names)
        self.fill_outputs(output, final_pred, pred_uncert)
        assert output.is_complete
        return output

    def fill_outputs(
        self,
        output: MarigoldIIDOutput,
        final_pred: torch.Tensor,
        pred_uncert: Optional[torch.Tensor] = None,
    ):
        for i, name in enumerate(self.target_names):
            start_idx = i * 3
            end_idx = start_idx + 3
            output.fill_entry(
                name=name,
                prediction=final_pred[:, start_idx:end_idx],
                uncertainty=(
                    pred_uncert[:, start_idx:end_idx]
                    if pred_uncert is not None
                    else None
                ),
                target_properties=self.target_properties,
            )

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if "trailing" != self.scheduler.config.timestep_spacing:
                logging.warning(
                    f"The loaded `DDIMScheduler` is configured with `timestep_spacing="
                    f'"{self.scheduler.config.timestep_spacing}"`; the recommended setting is `"trailing"`. '
                    f"This change is backward-compatible and yields better results. "
                    f"Consider using `prs-eth/marigold-iid-appearance-v1-1` or `prs-eth/marigold-iid-lighting-v1-1` "
                    f"for the best experience."
                )
            else:
                if n_step > 10:
                    logging.warning(
                        f"Setting too many denoising steps ({n_step}) may degrade the prediction; consider relying on "
                        f"the default values."
                    )
            if not self.scheduler.config.rescale_betas_zero_snr:
                logging.warning(
                    f"The loaded `DDIMScheduler` is configured with `rescale_betas_zero_snr="
                    f"{self.scheduler.config.rescale_betas_zero_snr}`; the recommended setting is True. "
                    f"Consider using `prs-eth/marigold-iid-appearance-v1-1` or `prs-eth/marigold-iid-lighting-v1-1` "
                    f"for the best experience."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            raise RuntimeError(
                "This pipeline implementation does not support the LCMScheduler. Please refer to the project "
                "README.md for instructions about using LCM."
            )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
    ) -> torch.Tensor:
        """
        Perform a single prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted targets of shape (B,3*n_targets,H,W).
        """
        device = self.device
        rgb_in = rgb_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)  # [B, 4, h, w]

        target_latent_shape = list(rgb_latent.shape)
        target_latent_shape[1] *= self.n_targets

        # Noisy latent for outputs
        target_latent = torch.randn(
            target_latent_shape, device=device, dtype=self.dtype, generator=generator
        )  # [B, 4*n_targets, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(
            device
        )  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, target_latent], dim=1
            )  # this order is important

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4*n_targets, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            target_latent = self.scheduler.step(
                noise_pred, t, target_latent, generator=generator
            ).prev_sample

        targets = self.decode_targets(target_latent)  # [B,3*n_targets,H,W]

        # clip prediction
        targets = torch.clip(targets, -1.0, 1.0)
        # shift to [0, 1]
        targets = (targets + 1.0) / 2.0

        return targets

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.latent_scale_factor
        return rgb_latent

    def decode_targets(self, target_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode target latents into image space.

        Args:
            target_latent: Target latent tensor of shape [B, 4*n_targets, h, w]

        Returns:
            Decoded target tensor of shape [B, 3*n_targets, H, W]
        """
        target_latent = target_latent / self.latent_scale_factor
        targets = []
        for i in range(self.n_targets):
            latent = target_latent[:, i * 4 : (i + 1) * 4, :, :]
            z = self.vae.post_quant_conv(latent)
            stacked = self.vae.decoder(z)
            targets.append(stacked)
        return torch.cat(targets, dim=1)
