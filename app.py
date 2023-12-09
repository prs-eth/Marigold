import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.model.marigold_pipeline import MarigoldPipeline
from src.util.ensemble import ensemble_depths
from src.util.image_util import chw2hwc, colorize_depth_maps, resize_max_res
from src.util.seed_all import seed_all
import gradio as gr

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

# Device
cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")
print(f"device = {device}")

# -------------------- Model --------------------
model = MarigoldPipeline.from_pretrained("Bingxin/Marigold")
model = model.to(device)


def infer_depth(
    input_image_path,
    resize_input=None,
    n_infer=10,
    denoise_steps=10,
    seed=None,
    merging_max_res=None,
    regularizer_strength=0.02,
    max_iter=5,
    reduction_method="median",
    tol=1e-3,
    depth_cmap="Spectral",
):
    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        # Read input image
        input_image = Image.open(input_image_path)

        # Resize image
        if resize_input is not None:
            image = resize_max_res(input_image, max_edge_resolution=resize_input)
        else:
            image = input_image

        # Convert the image to RGB, to 1.remove the alpha channel 2.convert B&W to 3-channel
        image = image.convert("RGB")

        image = np.asarray(image)

        # Normalize rgb values
        rgb = np.transpose(image, (2, 0, 1))  # [H, W, rgb] -> [rgb, H, W]
        rgb_norm = rgb / 255.0
        rgb_norm = torch.from_numpy(rgb_norm).unsqueeze(0).float()
        rgb_norm = rgb_norm.to(device)
        assert rgb_norm.min() >= 0.0 and rgb_norm.max() <= 1.0

        # Predict depth maps
        model.unet.eval()
        depth_pred_ls = []
        for _ in tqdm(range(n_infer), desc="multiple inference", leave=False):
            depth_pred_raw = model.forward(
                rgb_norm, num_inference_steps=denoise_steps, init_depth_latent=None
            )
            # clip prediction
            depth_pred_raw = torch.clip(depth_pred_raw, -1.0, 1.0)
            depth_pred_ls.append(depth_pred_raw.detach().cpu().numpy().copy())

        depth_preds = np.concatenate(depth_pred_ls, axis=0).squeeze()

        # Test-time ensembling
        if n_infer > 1:
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
        if resize_input is not None:
            pred_img = Image.fromarray(depth_pred)
            pred_img = pred_img.resize(input_image.size)
            depth_pred = np.asarray(pred_img)

        # scale prediction to [0, 1] for saving
        min_d = np.min(depth_pred)
        max_d = np.max(depth_pred)
        depth_to_save = (depth_pred - min_d) / (max_d - min_d)
        depth_to_save = (depth_to_save * 65535.0).astype(np.uint16)

        # Colorize depth map
        percentile = 0.03
        min_depth_pct = np.percentile(depth_pred, percentile)
        max_depth_pct = np.percentile(depth_pred, 100 - percentile)
        depth_colored = colorize_depth_maps(
            depth_pred, min_depth_pct, max_depth_pct, cmap=depth_cmap
        ).squeeze()  # [3, H, W], value in (0, 1)
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        Image.fromarray(depth_colored_hwc)

        return depth_colored_hwc, depth_to_save


def main():
    with gr.Blocks(title="Marigold Depth Estimation", analytics_enabled=False) as demo:
        gr.Markdown(
            """
            <p align="center">
            <a title="Website" href="https://marigoldmonodepth.github.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
            </a>
            <a title="arXiv" href="https://arxiv.org/abs/2312.02145" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
            </a>
            <a title="Github" href="https://github.com/prs-eth/marigold" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://img.shields.io/github/stars/prs-eth/marigold?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
            </a>
            <a title="Social" href="https://twitter.com/antonobukhov1" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
            </a>
            </p>
            <p align="justify">
            Marigold is the new state-of-the-art depth estimator for images in the wild. Upload your image into the pane on the left side, or expore examples listed in the bottom.  
            </p>
            """
        )
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image",
                    type="filepath",
                )
                submit_btn = gr.Button(value="Submit", variant="primary")
                with gr.Accordion("Advanced options", open=False):
                    resize_option = gr.Radio(
                        [None, 768], label="Resize input image", value=None
                    )
                    n_infer = gr.Slider(
                        label="Number of inference",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=10,
                    )
                    denoise_steps = gr.Slider(
                        label="Number of denoise steps",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=10,
                    )
                    seed = gr.Slider(
                        label="Random seed",
                        minimum=0,
                        maximum=100000,
                        step=1,
                        value=None,
                    )
                    merging_max_res = gr.Radio(
                        [None, 256], label="Merging max resolution", value=None
                    )
                    regularizer_strength = gr.Slider(
                        label="Regularizer strength",
                        minimum=0.0,
                        maximum=0.1,
                        step=0.01,
                        value=0.02,
                    )
                    max_iter = gr.Slider(
                        label="Max iteration",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=5,
                    )
                    reduction_method = gr.Radio(
                        ["median", "mean"], label="Reduction method", value="median"
                    )
                    tol = gr.Slider(
                        label="Tolerance",
                        minimum=1e-4,
                        maximum=1e-2,
                        step=1e-4,
                        value=1e-3,
                    )
            with gr.Column():
                depth_gallery = gr.Gallery()
        
        submit_btn.click(
            fn=infer_depth,
            inputs=[
                input_image,
                resize_option,
                n_infer,
                denoise_steps,
                seed,
                merging_max_res,
                regularizer_strength,
                max_iter,
                reduction_method,
                tol,
            ],
            outputs=depth_gallery,
        )

    demo.queue().launch(share=False)


if __name__ == "__main__":
    main()
