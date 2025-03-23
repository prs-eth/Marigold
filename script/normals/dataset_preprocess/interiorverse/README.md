# Interiorverse preprocessing

## Download

Download the [InteriorVerse](https://interiorverse.github.io/#download) dataset, you will need to apply for access first following the instructions on their webpage.

## Process dataset

Run the preprocessing script:

```bash
export BASE_DATA_DIR=<YOUR_DATA_DIR>  # Set target data directory

python script/dataset_preprocess_normals/interiorverse/preprocess_interiorverse_normals.py --dataset_dir path_to_downloaded_interiorverse --output_dir output_dir ${BASE_DATA_DIR}/interiorverse
```

Note that we only use the `scenes_85` data and not the `scenes_120` data.
