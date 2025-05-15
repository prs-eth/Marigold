#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"prs-eth/marigold-depth-v1-0"}
subfolder=${2:-"eval"}

python script/depth/infer.py \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir $BASE_DATA_DIR \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --dataset_config config/dataset_depth/data_diode_all.yaml \
    --output_dir output/${subfolder}/diode/prediction \
    --processing_res 640 \
    --resample_method bilinear
