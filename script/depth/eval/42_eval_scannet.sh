#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python script/depth/eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset_depth/data_scannet_val.yaml \
    --alignment least_square \
    --prediction_dir output/${subfolder}/scannet/prediction \
    --output_dir output/${subfolder}/scannet/eval_metric
