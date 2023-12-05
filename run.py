# Script for inference on (in-the-wild) images

# Author: Bingxin Ke
# Last modified: 2023-12-05


import argparse
import os
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.model.marigold_pipeline import MarigoldPipeline
from src.util.ensemble import ensemble_depths
from src.util.image_util import chw2hwc, colorize_depth_maps, resize_max_res
from src.util.seed_all import seed_all


EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


if "__main__" == __name__:
    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Run single-image depth estimation using Marigold.")
    parser.add_argument("--checkpoint", type=str, default="Bingxin/Marigold", help="Checkpoint path or hub name.")

    parser.add_argument("--input_rgb_dir", type=str, required=True, help="Path to the input image folder.")

    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")

    # resolution setting
    parser.add_argument("--resize_to_max_res", type=int, default=768, help="Resize the input image a max width/height while keeping aspect ratio. Only work when --resize_input is applied. Default: 768.")
    parser.add_argument("--not_resize_input", action="store_true", help="Use the original input resolution. Default: False.")
    parser.add_argument("--not_resize_output", action="store_true", help="When input is resized, out put depth at resized resolution. Default: False.")

    # inference setting
    parser.add_argument("--n_infer", type=int, default=10, help="Number of inferences to be ensembled, more inference gives better results but runs slower.")
    parser.add_argument("--denoise_steps", type=int, default=10, help="Inference denoising steps, more stepts results in higher accuracy but slower inference speed.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    # test-time ensembling
    parser.add_argument("--merging_max_res", type=int, default=None, help="Ensembling parameter, max resolution when ensembling, suggest setting to <800.")
    parser.add_argument("--regularizer_strength", type=float, default=0.02, help="Ensembling parameter, weight of optimization regularizer.")
    parser.add_argument("--tol", type=float, default=1e-3, help="Ensembling parameter, error tolorance.")
    parser.add_argument("--reduction_method", choices=["mean", "median"], default="median", help="Ensembling parameter, method to merge aligned depth maps.")
    parser.add_argument("--max_iter", type=int, default=5, help="Ensembling parameter, max optimization iterations.")
    
    # depth map colormap
    parser.add_argument("--depth_cmap", type=str, default="Spectral", help="Colormap used to render depth predictions.")

    args = parser.parse_args()

    checkpoint_path = args.checkpoint

    input_rgb_dir = args.input_rgb_dir

    output_dir = args.output_dir

    resize_to_max_res = args.resize_to_max_res
    resize_input = ~args.not_resize_input
    resize_back = ~args.not_resize_output if resize_input else False

    n_repeat = args.n_infer
    assert n_repeat >= 1
    denoise_steps = args.denoise_steps
    seed = args.seed

    merging_max_res = args.merging_max_res
    regularizer_strength = args.regularizer_strength
    max_iter = args.max_iter
    reduction_method = args.reduction_method
    tol = args.tol

    depth_cmap = args.depth_cmap

    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    # Output directories
    output_dir_color = os.path.join(output_dir, "depth_colored")
    output_dir_png = os.path.join(output_dir, "depth_bw")
    output_dir_npy = os.path.join(output_dir, "depth_npy")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_png, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    print(f"output dir: {output_dir}")

    # Device
    cuda_avail = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_avail else "cpu")
    print(f"device = {device}")

    # -------------------- Model --------------------
    model = MarigoldPipeline.from_pretrained(checkpoint_path)

    model = model.to(device)
    
    # -------------------- Data --------------------
    rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    print(f"Found {len(rgb_filename_list)} images")

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        for rgb_path in tqdm(rgb_filename_list, desc=f"Estimating depth", leave=True):
            # Read input image
            input_image = Image.open(rgb_path)

            # Resize image
            if resize_input:
                image = resize_max_res(
                    input_image, max_edge_resolution=resize_to_max_res
                )

            image = np.asarray(image)

            # Copy channels for B&W images
            if 2 == len(image.shape):
                image = image[:, :, np.newaxis]
                image = np.repeat(image, 3, axis=-1)

            # Normalize rgb values
            rgb = np.transpose(image, (2, 0, 1))  # [H, W, rgb] -> [rgb, H, W]
            rgb_norm = rgb / 255.0
            rgb_norm = torch.from_numpy(rgb_norm).unsqueeze(0).float()
            rgb_norm = rgb_norm.to(device)
            assert rgb_norm.min() >= 0.0 and rgb_norm.max() <= 1.0

            # Predict depth maps
            model.unet.eval()
            depth_pred_ls = []
            for i_rep in tqdm(range(n_repeat), desc="multiple inference", leave=False):
                depth_pred_raw = model.forward(
                    rgb_norm, num_inference_steps=denoise_steps, init_depth_latent=None
                )
                # clip prediction
                depth_pred_raw = torch.clip(depth_pred_raw, -1.0, 1.0)
                depth_pred_ls.append(depth_pred_raw.detach().cpu().numpy().copy())

            depth_preds = np.concatenate(depth_pred_ls, axis=0).squeeze()

            # Test-time ensembling
            if n_repeat > 1:
                depth_pred, pred_uncert = ensemble_depths(
                    depth_preds,
                    regularizer_strength=regularizer_strength,
                    max_iter=max_iter,
                    tol=tol,
                    reduction=reduction_method,
                    max_res=merging_max_res,
                    device=device,
                )
            else:
                depth_pred = depth_preds

            # Resize back to original resolution
            if resize_back:
                pred_img = Image.fromarray(depth_pred)
                pred_img = pred_img.resize(input_image.size)
                depth_pred = np.asarray(pred_img)

            # Save as npy
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_pred"
            save_to_npy = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            np.save(save_to_npy, depth_pred)

            # Save as 16uint png
            save_to_png = os.path.join(output_dir_png, f"{pred_name_base}.png")
            # scale prediction to [0, 1]
            min_d = np.min(depth_pred)
            max_d = np.max(depth_pred)
            depth_to_save = (depth_pred - min_d) / (max_d - min_d)
            depth_to_save = (depth_to_save * 65535.0).astype(np.uint16)
            cv2.imwrite(save_to_png, depth_to_save)

            # Colorize
            percentile = 0.03
            min_depth_pct = np.percentile(depth_pred, percentile)
            max_depth_pct = np.percentile(depth_pred, 100 - percentile)
            save_to_color = os.path.join(
                output_dir_color, f"{pred_name_base}_colored.png"
            )
            depth_colored = colorize_depth_maps(
                depth_pred, min_depth_pct, max_depth_pct, cmap=depth_cmap
            ).squeeze()  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            Image.fromarray(depth_colored_hwc).save(save_to_color)
