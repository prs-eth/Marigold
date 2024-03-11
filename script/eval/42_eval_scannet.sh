#!/usr/bin/env bash
set -e
set -x


python eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset/data_scannet_val.yaml \
    --alignment least_square \
    --prediction_dir output/scannet/prediction \
    --output_dir output/scannet/eval_metric \
