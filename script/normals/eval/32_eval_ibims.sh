#!/usr/bin/env bash
# set -e
set -x

subfolder=${1:-"eval"}

python script/normals/eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset_normals/data_ibims_test.yaml \
    --prediction_dir output/${subfolder}/ibims_normals_test/prediction \
    --output_dir output/${subfolder}/ibims_normals_test/eval_metric
