#!/usr/bin/env bash
set -e
set -x


python infer.py \
    --seed 1234 \
    --base_data_dir $BASE_DATA_DIR \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --dataset_config config/dataset/data_diode_all.yaml \
    --output_dir output/diode/prediction \
    --processing_res 640 \
    --resample_method bilinear \
