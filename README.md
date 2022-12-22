# AnomalyBERT: Transformer-based Anomaly Detector

This is the code for **Self-supervised Transformer for Time Series Anomaly Detection**.

## Installation

Please clone our repository and install the packages in `requirements.txt`.
Before installing the packages, we recommend installing Python 3.8 and Pytorch 1.9 with CUDA.

```
cd path/to/repository/
git clone https://github.com/Jhryu30/AnomalyBERT.git

conda create --name your_env_name python=3.8
conda activate your_env_name

pip install torch==1.9.0+cuXXX -f https://download.pytorch.org/whl/torch_stable.html  # cuXXX for your CUDA setting
pip install -r requirements.txt
```

We use SMAP public dataset from NASA. You can download the dataset and preprocess it.
(데이터 전처리 파일 미완성이므로 아직 사용 X. 일단 전처리된 데이터는 google drive 같은 곳에 올릴 예정)

```
cd path/to/data/
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip
unzip data.zip

wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
mv labeled_anomalies.csv data/

cd path/to/repository/
python3 utils/data_preprocessing.py --dataset=SMAP --data_dir=path/to/data/
```

After preprocessing, you need to edit your dataset directory in `utils/config.py`.

```
DATASET_DIR = 'path/to/data/'
```

## Training

We provide the training code for our model.
Before training, you need to create an empty folder for log files.

```
mkdir logs  # Create log folder once.
```

For example, to train a model of 6-layer Transformer body on SMAP dataset, run:

```
python3 train.py --dataset=SMAP --n_layer=6
```

To train a model on MSL dataset with patch size of 2 and customized outlier synthesis probability, run:

```
python3 train.py --dataset=MSL --patch_size=2 --soft_replacing=0.5 --uniform_replacing=0.1 --peak_noising=0.1 \
--length_adjusting=0.1
```

If you want to customize the model and training settings, please check the options in `train.py`.

## Anomaly score estimation and metric computation

To estimate anomaly scores of test data with the trained model, run the `estimate.py` code.
For example, you can estimate anomaly scores of SMAP test set divided by channel with window sliding of 16.

```
python3 estimate.py --dataset=SMAP --model=logs/YYMMDDhhmmss_SMAP/model.pt --state_dict=logs/YYMMDDhhmmss_SMAP/state_dict.pt \
--data_division=channel --window_sliding=16
```

Now you will obtain results (npy) file that contains the estimated anomaly scores.
With the results file, you can compute F1-score with and without the point adjustment by running:

```
python3 compute_metrics.py --dataset=SMAP --result=logs/YYMMDDhhmmss_SMAP/state_dict_results.npy
```

If you want to customize the estimation or computation settings, please check the options in `estimate.py` and `compute_metrics.py`.