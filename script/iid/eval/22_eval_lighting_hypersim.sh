#!/usr/bin/env bash
# set -e
set -x

subfolder=${1:-"eval"}

python script/iid/eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset_iid/data_lighting_hypersim_test.yaml \
    --prediction_dir output/${subfolder}/iid_lighting_hypersim_test/prediction \
    --output_dir output/${subfolder}/iid_lighting_hypersim_test/eval_metric \
    --target_names albedo shading residual \
    --use_mask
