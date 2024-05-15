# Hypersim preprocessing

## Download

Download [Hypersim](https://github.com/apple/ml-hypersim) dataset using [this script](https://github.com/apple/ml-hypersim/blob/20f398f4387aeca73175494d6a2568f37f372150/code/python/tools/dataset_download_images.py).

Download the scene split file from [here](https://github.com/apple/ml-hypersim/blob/main/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv).

## Process dataset

Run the preprocessing script:

```bash
python script/dataset_preprocess/hypersim/preprocess_hypersim.py --split_csv /path/to/metadata_images_split_scene_v1.csv
```

(optional) Tar the processed data, for example:

```bash
cd data/Hypersim/processed/train
tar -cf ../../hypersim_processed_train.tar .
```
