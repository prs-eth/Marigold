#!/usr/bin/env bash
set -e
set -x

ckpt_dir=${ckpt_dir:-checkpoint}
mkdir -p $ckpt_dir
cd $ckpt_dir

checkpoint_name=$1

if [ -d $checkpoint_name ]; then
    exit 0
fi

if [[ "$checkpoint_name" == *"normals"* || "$checkpoint_name" == *"iid"* ]]; then
    wget -nv --show-progress https://share.phys.ethz.ch/~pf/bingkedata/marigold/marigold_normals/checkpoint/${checkpoint_name}.zip
    unzip ${checkpoint_name}.zip
    rm -f ${checkpoint_name}.zip

else
    wget -nv --show-progress https://share.phys.ethz.ch/~pf/bingkedata/marigold/checkpoint/${checkpoint_name}.tar
    tar -xf ${checkpoint_name}.tar
    rm -f ${checkpoint_name}.tar
fi

