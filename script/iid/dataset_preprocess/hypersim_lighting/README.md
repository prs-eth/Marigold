# Hypersim IID-Lighting Preprocessing

## Download

Download the [Hypersim](https://github.com/apple/ml-hypersim) dataset using [this script](https://github.com/apple/ml-hypersim/blob/20f398f4387aeca73175494d6a2568f37f372150/code/python/tools/dataset_download_images.py).

Download the scene split file from [here](https://github.com/apple/ml-hypersim/blob/main/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv).

## Preprocess
Run the preprocessing script:
```bash
export BASE_DATA_DIR=<YOUR_DATA_DIR>  # Set target data directory

python script/iid/dataset_preprocess/hypersim_lighting/preprocess_hypersim_iid.py --split_csv /path/to/metadata_images_split_scene_v1.csv --dataset_dir path_to_downloaded_hypersim --output_dir ${BASE_DATA_DIR}/hypersim
```
The preprocessing script filters out samples with invalid albedo, shading and residual values. For training our Marigold-IID-Lighting model, we used both filtered train and val split as training data (the complete filtered sample list can be found [here](../../../../data_split/hypersim_iid/hypersim_train_filtered.txt)). 


