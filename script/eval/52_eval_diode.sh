#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset/data_diode_all.yaml \
    --alignment least_square \
    --prediction_dir output/${subfolder}/diode/prediction \
    --output_dir output/${subfolder}/diode/eval_metric \
