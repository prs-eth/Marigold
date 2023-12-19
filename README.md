# Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation

This repository represents the official implementation of the paper titled "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation".

[![Website](doc/badges/badge-website.svg)](https://marigoldmonodepth.github.io)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2312.02145)
[![Open In Colab](doc/badges/badge-colab.svg)](https://colab.research.google.com/drive/12G8reD13DdpMie5ZQlaFNo2WCGeNUH-u?usp=sharing)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/toshas/marigold)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/Bingxin/Marigold)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)
<!-- [![Website](https://img.shields.io/badge/Project-Website-1081c2)](https://arxiv.org/abs/2312.02145) -->
<!-- [![GitHub](https://img.shields.io/github/stars/prs-eth/Marigold?style=default&label=GitHub%20★&logo=github)](https://github.com/prs-eth/Marigold) -->
<!-- [![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)]() -->
<!-- [![Docker](doc/badges/badge-docker.svg)]() -->

[Bingxin Ke](http://www.kebingxin.com/),
[Anton Obukhov](https://www.obukhov.ai/),
[Shengyu Huang](https://shengyuh.github.io/),
[Nando Metzger](https://nandometzger.github.io/),
[Rodrigo Caye Daudt](https://rcdaudt.github.io/),
[Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en )

We present Marigold, a diffusion model and associated fine-tuning protocol for monocular depth estimation. Its core principle is to leverage the rich visual knowledge stored in modern generative image models. Our model, derived from Stable Diffusion and fine-tuned with synthetic data, can zero-shot transfer to unseen data, offering state-of-the-art monocular depth estimation results.

![teaser](doc/teaser_collage_transparant.png)

## 📢 News

2023-12-19: Updated [license](LICENSE.txt) to Apache License, Version 2.0.<br>
2023-12-08: Added
<a href="https://huggingface.co/spaces/toshas/marigold"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow" height="16"></a> - try it out with your images for free!<br>
2023-12-05: Added <a href="https://colab.research.google.com/drive/12G8reD13DdpMie5ZQlaFNo2WCGeNUH-u?usp=sharing"><img src="doc/badges/badge-colab.svg" height="16"></a> - dive deeper into our inference pipeline!<br>
2023-12-04: Added <a href="https://arxiv.org/abs/2312.02145"><img src="https://img.shields.io/badge/arXiv-PDF-b31b1b" height="16"></a>
paper and inference code (this repository).

## 🚀 Usage

We offer several ways to interact with Marigold:

1. A free online interactive demo is available here: <a href="https://huggingface.co/spaces/toshas/marigold"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow" height="16"></a> (kudos to the HF team for the GPU grant)

2. Run the demo locally (requires a GPU and an `nvidia-docker2`, see [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)): `docker run -it -p 7860:7860 --platform=linux/amd64 --gpus all registry.hf.space/toshas-marigold:latest python app.py` 

3. Extended demo on a Google Colab: <a href="https://colab.research.google.com/drive/12G8reD13DdpMie5ZQlaFNo2WCGeNUH-u?usp=sharing"><img src="doc/badges/badge-colab.svg" height="16"></a>

4. If you just want to see the examples, visit our gallery: <a href="https://marigoldmonodepth.github.io"><img src="doc/badges/badge-website.svg" height="16"></a>

5. Finally, local development instructions are given below.

## 🛠️ Setup

This code was tested on:

- Ubuntu 22.04 LTS, Python 3.10.12,  CUDA 11.7, GeForce RTX 3090 (pip, Mamba)
- CentOS Linux 7, Python 3.10.4, CUDA 11.7, GeForce RTX 4090 (pip)
- Windows 11 22H2, Python 3.10.12, CUDA 12.3, GeForce RTX 3080 (Mamba)
- MacOS 14.2, Python 3.10.12, M1 16G (pip)

### 🪟 A Note for Windows users

We recommend running the code in WSL2:

1. Install WSL following [installation guide](https://learn.microsoft.com/en-us/windows/wsl/install#install-wsl-command).
1. Install CUDA support for WSL following [installation guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2).
1. Find your drives in `/mnt/<drive letter>/`; check [WSL FAQ](https://learn.microsoft.com/en-us/windows/wsl/faq#how-do-i-access-my-c--drive-) for more details. Navigate to the working directory of choice. 

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/prs-eth/Marigold.git
cd Marigold
```

### 💻 Dependencies

We provide several ways to install the dependencies.

1. **Using [Mamba](https://github.com/mamba-org/mamba)**, which can installed together with [Miniforge3](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3). 

    Windows users: Install the Linux version into the WSL.

    After the installation, Miniforge needs to be activated first: `source /home/$USER/miniforge3/bin/activate`.

    Create the environment and install dependencies into it:

    ```bash
    mamba env create -n marigold --file environment.yaml
    conda activate marigold
    ```

2. **Using pip:** 
    Alternatively, create a Python native virtual environment and install dependencies into it:

    ```bash
    python -m venv venv/marigold
    source venv/marigold/bin/activate
    pip install -r requirements.txt
    ```

Keep the environment activated before running the inference script. 
Activate the environment again after restarting the terminal session.

## 🚀 Testing on your images

### 📷 Prepare images

If you have images at hand, skip this step. Otherwise, download a few select images from our paper:
```bash
bash script/download_sample_data.sh
```

### 🎮 Run inference

Place your images in a directory, for example, under `input/in-the-wild_example`, and run the following command:

```bash
python run.py \
    --input_rgb_dir input/in-the-wild_example \
    --output_dir output/in-the-wild_example
```

You can find all results in `output/in-the-wild_example`. Enjoy!

### ⚙️ Inference settings

The default settings are optimized for the best result. However, the behavior of the code can be customized:

- Trade-offs between the **accuracy** and **speed** (for both options, larger values result in better accuracy at the cost of slower inference.)

  - `--ensemble_size`: Number of inference passes in the ensemble. Default: 10.
  - `--denoise_steps`: Number of denoising steps of each inference pass. Default: 10.

- `--half_precision`: Run with half-precision (16-bit float) to reduce VRAM usage, might lead to suboptimal result.

- By default, the inference script resizes input images to the *processing resolution*, and then resizes the prediction back to the original resolution. This gives the best quality, as Stable Diffusion, from which Marigold is derived, performs best at 768x768 resolution.  
  
  - `--processing_res`: the processing resolution; set 0 to process the input resolution directly. Default: 768.
  - `--output_processing_res`: produce output at the processing resolution instead of upsampling it to the input resolution. Default: False.

- `--seed`: Random seed can be set to ensure additional reproducibility. Default: None (using current time as random seed).
- `--batch_size`: Batch size of repeated inference. Default: 0 (best value determined automatically).
- `--color_map`: [Colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) used to colorize the depth prediction. Default: Spectral.
- `--apple_silicon`: Use Apple Silicon MPS acceleration.

### ⬇ Checkpoint cache

By default, the [checkpoint](https://huggingface.co/Bingxin/Marigold) is stored in the Hugging Face cache.
The `HF_HOME` environment variable defines its location and can be overridden:

```bash
export HF_HOME=new/path
```

Alternatively, use the following script to download the checkpoint weights locally:

```bash
bash script/download_weights.sh
```

At inference, specify the checkpoint path:

```bash
python run.py \
    --checkpoint checkpoint/Marigold_v1_merged_2 \
    --input_rgb_dir input/in-the-wild_example\
    --output_dir output/in-the-wild_example
```

## ✏️ Contributing

Please refer to [this](CONTRIBUTING.md) instruction.

## 🤔 Troubleshooting

| Problem                                                                                                                                      | Solution                                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| (Windows) Invalid DOS bash script on WSL                                                                                                     | Run `dos2unix <script_name>` to convert script format          |
| (Windows) error on WSL: `Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory` | Run `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` |


## 🎓 Citation

Please cite our paper:

```bibtex
@misc{ke2023repurposing,
      title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation}, 
      author={Bingxin Ke and Anton Obukhov and Shengyu Huang and Nando Metzger and Rodrigo Caye Daudt and Konrad Schindler},
      year={2023},
      eprint={2312.02145},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 🎫 License

This work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

By downloading and using the code and model you agree to the terms in the  [LICENSE](LICENSE.txt).

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)
