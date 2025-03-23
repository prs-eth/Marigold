#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"prs-eth/marigold-depth-v1-1"}
subfolder=${2:-"eval"}
n_ensemble=${3:-10}

python script/depth/infer.py \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir $BASE_DATA_DIR \
    --denoise_steps 1 \
    --ensemble_size ${n_ensemble} \
    --processing_res 0 \
    --dataset_config config/dataset_depth/data_scannet_val.yaml \
    --output_dir output/${subfolder}/scannet/prediction
