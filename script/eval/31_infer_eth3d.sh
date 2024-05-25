#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"prs-eth/marigold-v1-0"}
subfolder=${2:-"eval"}

python infer.py  \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir $BASE_DATA_DIR \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --dataset_config config/dataset/data_eth3d.yaml \
    --output_dir output/${subfolder}/eth3d/prediction \
    --processing_res 756 \
    --resample_method bilinear \