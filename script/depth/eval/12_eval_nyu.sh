#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python script/depth/eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset_depth/data_nyu_test.yaml \
    --alignment least_square \
    --prediction_dir output/${subfolder}/nyu_test/prediction \
    --output_dir output/${subfolder}/nyu_test/eval_metric
