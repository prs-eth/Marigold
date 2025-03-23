# InteriorVerse IID-Appearance Preprocessing

## Download

Download the [InteriorVerse](https://interiorverse.github.io/#download) dataset, you will need to apply for access first following the instructions on their webpage.

After download and unzip, the file structure should look like this:
```text
your/raw/InteriorVerse/
└── scenes_85/
   └── L3D124S8ENDIDQ4AKYUI5NGMLUF3P3WC888/
       ├── 000_albedo.exr
       ├── 000_depth.exr
       ├── 000_im.exr
       ├── 000_mask.exr
       ├── 000_material.exr
       ├── 000_normal.exr
       └── ...
```
Note that we only use scenes_85, scenes_120 is not needed.

## Preprocess
We only need `albedo`, `material`, `im` and `mask` images, use `preprocess_interiorverse_iid.py` to tar them:
```bash
python preprocess_interiorverse_iid.py --output_data_dir ${BASE_DATA_DIR} --raw_data_dir your/raw/InteriorVerse
```



