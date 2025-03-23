#!/usr/bin/env bash
# set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"prs-eth/marigold-iid-appearance-v1-1"}
subfolder=${2:-"eval"}

python script/iid/infer.py \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir $BASE_DATA_DIR \
    --denoise_steps 4 \
    --ensemble_size 1 \
    --processing_res 640 \
    --dataset_config config/dataset_iid/data_appearance_interiorverse_test.yaml \
    --output_dir output/${subfolder}/iid_appearance_interiorverse_test/prediction
