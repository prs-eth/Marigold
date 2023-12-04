#!/usr/bin/env bash
set -e
set -x

data_dir=data
mkdir $data_dir
cd $data_dir
wget -nv --show-progress https://share.phys.ethz.ch/~pf/bingkedata/marigold/in-the-wild_example.tar

tar -xf in-the-wild_example.tar
rm in-the-wild_example.tar
