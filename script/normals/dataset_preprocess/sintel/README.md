# Hypersim preprocessing

## Download

Download the preprocessed [Sintel](http://sintel.is.tue.mpg.de/) dataset as follows:

```bash
export BASE_DATA_DIR=<YOUR_DATA_DIR>  # Set target data directory

wget -r -np -nH --cut-dirs=4 -R "index.html*" -P ${BASE_DATA_DIR} https://share.phys.ethz.ch/~pf/bingkedata/marigold/marigold_normals/sintel.zip
unzip ${BASE_DATA_DIR}/sintel.zip -d ${BASE_DATA_DIR}/
rm -f ${BASE_DATA_DIR}/sintel.zip
```

For training, we filtered out low-quality samples and used the samples listed [here](../../../../data_split/sintel_normals/sintel_filtered.txt)

