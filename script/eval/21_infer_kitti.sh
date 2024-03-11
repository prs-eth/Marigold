#!/usr/bin/env bash
set -e
set -x


python infer.py \
    --seed 1234 \
    --base_data_dir $BASE_DATA_DIR \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --processing_res 0 \
    --dataset_config config/dataset/data_kitti_eigen_test.yaml \
    --output_dir output/kitti_eigen_test/prediction \
