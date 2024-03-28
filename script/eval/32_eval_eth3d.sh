#!/usr/bin/env bash
set -e
set -x


python eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset/data_eth3d.yaml \
    --alignment least_square \
    --prediction_dir  output/eth3d/prediction \
    --output_dir output/eth3d/eval_metric \
    --alignment_max_res 1024 \