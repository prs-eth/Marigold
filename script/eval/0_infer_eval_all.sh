#!/usr/bin/env bash
set -e
set -x

bash script/eval/11_infer_nyu.sh
bash script/eval/12_eval_nyu.sh

bash script/eval/21_infer_kitti.sh
bash script/eval/22_eval_kitti.sh

bash script/eval/31_infer_eth3d.sh
bash script/eval/32_eval_eth3d.sh

bash script/eval/41_infer_scannet.sh
bash script/eval/42_eval_scannet.sh

bash script/eval/51_infer_diode.sh
bash script/eval/52_eval_diode.sh