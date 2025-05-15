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

import cv2
import numpy as np
import tarfile
import torch
from PIL import Image
from io import BytesIO
from typing import Union


def img_hwc2chw(img: Union[np.ndarray, torch.Tensor]):
    assert len(img.shape) == 3
    if isinstance(img, np.ndarray):
        return np.transpose(img, (2, 0, 1))
    if isinstance(img, torch.Tensor):
        return img.permute(2, 0, 1)
    raise TypeError("img should be np.ndarray or torch.Tensor")


def img_chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    else:
        raise TypeError("img should be np.ndarray or torch.Tensor")
    return hwc


def img_int2float(img, dtype=None):
    if dtype is not None:
        if isinstance(img, np.ndarray):
            img = img.astype(dtype)
        else:
            img = img.to(dtype)
    return img / 255.0


def img_float2int(img):
    if isinstance(img, np.ndarray):
        return (img * 255.0).astype(np.uint8)
    else:
        return (img * 255.0).to(torch.uint8)


def img_normalize(img):
    return img * 2.0 - 1.0


def img_denormalize(img):
    return img * 0.5 + 0.5


def img_linear2srgb(img):
    return img ** (1 / 2.2)


def img_srgb2linear(img):
    return img**2.2


def write_img(img: np.ndarray, path):
    img = img_float2int(img)
    if len(img.shape) == 3:
        img = img[:, :, ::-1]  # RGB->BGR
    cv2.imwrite(path, img)


def _read_image_from_buffer(buffer: BytesIO, is_hdr: bool) -> np.ndarray:
    if is_hdr:
        file_bytes = np.frombuffer(buffer.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.clip(img, 0, 1)
    else:
        img = Image.open(buffer)  # [H, W, rgb]
        img = np.asarray(img)
        img = img_int2float(img)

    return img


def is_hdr(path: str):
    return path.endswith(".exr")


def read_img_from_tar(tar_file: tarfile.TarFile, rel_path: str) -> np.ndarray:
    tar_obj = tar_file.extractfile(rel_path)
    buffer = BytesIO(tar_obj.read())
    img = _read_image_from_buffer(buffer, is_hdr(rel_path))
    return img


def read_img_from_file(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        buffer = BytesIO(f.read())
        img = _read_image_from_buffer(buffer, is_hdr(path))
    return img
