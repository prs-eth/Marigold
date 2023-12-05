#!/usr/bin/env bash
set -e
set -x

ckpt_dir=${ckpt_dir:-checkpoint}
mkdir $ckpt_dir
cd $ckpt_dir
wget -nv --show-progress https://share.phys.ethz.ch/~pf/bingkedata/marigold/Marigold_v1_merged.tar

tar -xf Marigold_v1_merged.tar
rm Marigold_v1_merged.tar
