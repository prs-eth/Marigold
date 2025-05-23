#!/usr/bin/env bash
set -e
set -x

data_dir=input
mkdir -p $data_dir
cd $data_dir

if test -f "in-the-wild_example2.tar" ; then
    echo "Tar file exists: in-the-wild_example2.tar"
    exit 1
fi

wget -nv --show-progress https://share.phys.ethz.ch/~pf/bingkedata/marigold/in-the-wild_example2.tar

tar -xf in-the-wild_example2.tar
rm in-the-wild_example2.tar
