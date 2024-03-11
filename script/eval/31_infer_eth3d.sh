#!/usr/bin/env bash
set -e
set -x


python infer.py \
    --seed 1234 \
    --base_data_dir $BASE_DATA_DIR \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --dataset_config config/dataset/data_eth3d.yaml \
    --output_dir output/eth3d/prediction \
    --processing_res 756 \
    --resample_method bilinear \