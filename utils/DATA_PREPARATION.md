# Data Preparation for AnomalyBERT

This is an instruction for preprocessing of five benchmark datasets; SMAP, MSL, SMD, SWaT, and WADI.
Before you download dataset files, please set your dataset folder `path/to/dataset/` and then follow the instructions below.

## Data Download
### SMAP & MSL

```
cd path/to/dataset/
mkdir telemanom && cd telemanom
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip
unzip data.zip

wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
mv labeled_anomalies.csv data/
```

### SMD

```
cd path/to/dataset/
git clone https://github.com/smallcowbaby/OmniAnomaly.git
```

### SWaT & WADI

To download SWaT and WADI datasets, you have to request access to the files at [here](https://itrust.sutd.edu.sg/itrust-labs_datasets/).
After you receive the download links, please create folders for both datasets `SWaT/` and `WADI/`, and download "SWaT.A1 & A2_Dec 2015/Physical" folder for SWaT and "WADI.A2_19 Nov 2019" folder for WADI.

The structure of your dataset directory should be as follows:

```
path/to/dataset/
|-- SWaT
    |-- Physical
        |-- ...
|-- WADI
    |-- WADI.A2_19 Nov 2019
        |-- ...
|-- processed
|-- ...
```

## Data Preprocessing

If you finish downloading the entire (or a part of) datasets, you need to preprocess them by running the command below.

```
cd path/to/repository/
python3 utils/data_preprocessing.py --data_dir=path/to/dataset/ --dataset=SMAP # or one of the dataset names MSL/SMD/SWaT/WADI
```

After preprocessing, you will have three npy files `{dataset}_train.npy`, `{dataset}_test.npy`, and `{dataset}_test_label.npy` in the `path/to/dataset/processed/` folder.
Please write down the path to you processed dataset directory in `utils/config.py` as below.

```
DATASET_DIR = 'path/to/dataset/processed/'
```