# Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation

This repository represents the official implementation of the paper titled "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation".

[![Website](doc/badges/badge-website.svg)](https://marigoldmonodepth.github.io)
[![Paper](doc/badges/badge-pdf.svg)](https://arxiv.org/abs/2312.02145)
[![HF Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/Bingxin/Marigold)
[![License](doc/badges/badge-license.svg)](LICENSE)
<!-- [![GitHub](https://img.shields.io/github/stars/prs-eth/Marigold?style=default&label=GitHub%20★&logo=github)](https://github.com/prs-eth/Marigold) -->
<!-- [![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)]() -->
<!-- [![Open In Colab](doc/badges/badge-colab.svg)]() -->
<!-- [![Docker](doc/badges/badge-docker.svg)]() -->
<!-- ### [Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation]() -->

[Bingxin Ke](http://www.kebingxin.com/),
[Anton Obukhov](https://www.obukhov.ai/),
[Shengyu Huang](https://shengyuh.github.io/),
[Nando Metzger](https://nandometzger.github.io/),
[Rodrigo Caye Daudt](https://rcdaudt.github.io/),
[Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en )

We present Marigold, a diffusion model and associated fine-tuning protocol for monocular depth estimation. Its core principle is to leverage the rich visual knowledge stored in modern generative image models. Our model, derived from Stable Diffusion and fine-tuned with synthetic data, can zero-shot transfer to unseen data, offering state-of-the-art monocular depth estimation results.

![teaser](doc/teaser_collage_transparant.png)

## 📢 News

2023-12-04: Upload [arxiv](https://arxiv.org/abs/2312.02145) paper.

2023-12-04: Release inference code.

## 🛠️ Setup

This code has been tested on:

- Python 3.10.12, PyTorch 2.0.1, CUDA 11.7, GeForce RTX 3090
- python 3.10.4, Pytorch 2.0.1, CUDA 11.7, GeForce RTX 4090


### 💻 Dependencies

```bash
python -m venv venv/marigold
source venv/marigold/bin/activate
pip install -r requirements.txt
```

## 🚀 Inference on in-the-wild images

### 📷 Sample images

```bash
bash script/download_sample_data.sh
```

### 🎮 Inference

This script will automatically download the [checkpoint](https://huggingface.co/Bingxin/Marigold).

```bash
python run.py \
    --input_rgb_dir data/in-the-wild_example\
    --output_dir output/in-the-wild_example
```

### ⚙️ Inference settings

- The inference script by default will resize the input images and resize back to the original resolution.
  
  - `--resize_to_max_res`: The maximum edge length of resized input image. Default: 768.
  - `--not_resize_input`: If given, will not resize the input image.
  - `--not_resize_output`: If given, will not resize the output image back to the original resolution. Only valid without `--not_resize_input` option.

- Trade-offs between **accuracy** and **speed** (for both options, larger value results in more accurate results at the cost of slower inference speed.)

  - `--n_infer`: Number of inference passes to be ensembled. Default: 10.
  - `--denoise_steps`: Number of diffusion denoising steps of each inference pass. Default: 10.

- `--seed`: Random seed, can be set to ensure reproducibility. Default: None (using current time as random seed).
- `--depth_cmap`: Colormap used to colorize the depth prediction. Default: Spectral.

- The model cache directory can be controlled by environment variable `HF_HOME`, for example:

    ```bash
    export HF_HOME=$(pwd)/checkpoint
    ```

### ⬇ Using local checkpoint

```bash
# Download checkpoint
bash script/download_weights.sh
```

```bash
python run.py \
    --checkpoint checkpoint/Marigold_v1_merged \
    --input_rgb_dir data/in-the-wild_example\
    --output_dir output/in-the-wild_example
```

## 🎓 Citation

```bibtex
@misc{ke2023marigold,
  author    = {Ke, Bingxin and Obukhov, Anton and Huang, Shengyu and Metzger, Nando and Daudt, Rodrigo Caye and Schindler, Konrad},
  title     = {Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
  year      = {2023},
}
```

## License

This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

[<img src="doc/badges/badge-license.svg" height="20"/>](http://creativecommons.org/licenses/by-nc-sa/4.0/)