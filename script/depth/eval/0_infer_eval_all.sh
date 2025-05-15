#!/usr/bin/env bash
set -e
set -x

n_ensemble=1
ckpt="prs-eth/marigold-depth-v1-1"
subfolder="eval_1step_${n_ensemble}ensemble"

bash script/depth/eval/11_infer_nyu.sh ${ckpt} ${subfolder} ${n_ensemble}
bash script/depth/eval/12_eval_nyu.sh ${subfolder}

bash script/depth/eval/21_infer_kitti.sh ${ckpt} ${subfolder} ${n_ensemble}
bash script/depth/eval/22_eval_kitti.sh ${subfolder}

bash script/depth/eval/31_infer_eth3d.sh ${ckpt} ${subfolder} ${n_ensemble}
bash script/depth/eval/32_eval_eth3d.sh ${subfolder}

bash script/depth/eval/41_infer_scannet.sh ${ckpt} ${subfolder} ${n_ensemble}
bash script/depth/eval/42_eval_scannet.sh ${subfolder}

bash script/depth/eval/51_infer_diode.sh ${ckpt} ${subfolder} ${n_ensemble}
bash script/depth/eval/52_eval_diode.sh ${subfolder}
