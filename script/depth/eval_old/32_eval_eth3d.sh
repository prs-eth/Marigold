#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python script/depth/eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset_depth/data_eth3d.yaml \
    --alignment least_square \
    --prediction_dir  output/${subfolder}/eth3d/prediction \
    --output_dir output/${subfolder}/eth3d/eval_metric \
    --alignment_max_res 1024
