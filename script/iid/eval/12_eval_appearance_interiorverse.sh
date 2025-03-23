#!/usr/bin/env bash
# set -e
set -x

subfolder=${1:-"eval"}

python script/iid/eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset_iid/data_appearance_interiorverse_test.yaml \
    --prediction_dir output/${subfolder}/iid_appearance_interiorverse_test/prediction \
    --output_dir output/${subfolder}/iid_appearance_interiorverse_test/eval_metric \
    --target_names albedo material \
    --targets_to_eval_in_linear_space material