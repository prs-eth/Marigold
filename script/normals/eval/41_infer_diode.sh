#!/usr/bin/env bash
# set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"prs-eth/marigold-normals-v1-1"}
subfolder=${2:-"eval"}

python script/normals/infer.py \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir $BASE_DATA_DIR \
    --denoise_steps 4 \
    --ensemble_size 10 \
    --processing_res 768 \
    --dataset_config config/dataset_normals/data_diode_test.yaml \
    --output_dir output/${subfolder}/diode_normals_test/prediction
