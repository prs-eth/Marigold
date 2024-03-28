#!/usr/bin/env bash
set -e
set -x


python eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset/data_diode_all.yaml \
    --alignment least_square \
    --prediction_dir output/diode/prediction \
    --output_dir output/diode/eval_metric \
