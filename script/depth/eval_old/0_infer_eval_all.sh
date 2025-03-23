#!/usr/bin/env bash
set -e
set -x

ckpt=${1:-"prs-eth/marigold-depth-v1-0"}

bash script/depth/eval_old/11_infer_nyu.sh ${ckpt}
bash script/depth/eval_old/12_eval_nyu.sh

bash script/depth/eval_old/21_infer_kitti.sh ${ckpt}
bash script/depth/eval_old/22_eval_kitti.sh

bash script/depth/eval_old/31_infer_eth3d.sh ${ckpt}
bash script/depth/eval_old/32_eval_eth3d.sh

bash script/depth/eval_old/41_infer_scannet.sh ${ckpt}
bash script/depth/eval_old/42_eval_scannet.sh

bash script/depth/eval_old/51_infer_diode.sh ${ckpt}
bash script/depth/eval_old/52_eval_diode.sh
