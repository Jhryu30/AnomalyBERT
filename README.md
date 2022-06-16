# Transformer-based Anomaly Detector

This is the code for **Self-supervised Transformer for Time Series Anomaly Detection**.

## Installation

Please clone our repository and install the packages in `requirements.txt`.
Before installing the packages, we recommend installing Python 3.8 and Pytorch 1.9.

```
cd path/to/repository/
git clone https://github.com/coffeetumbler/TransformerBasedAnomalyDetector.git

conda create --name your_env_name python=3.8
conda activate your_env_name

conda install pytorch==1.9.0 -c pytorch
pip install -r requirements.txt
```

We use SMAP public dataset from NASA. You can download the dataset and preprocess it.

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
python3 train_smap.py
```

To customize the model and training setting, please check the options in `train_smap.py`.

## Evaluation

You can compute our anomaly indices on a trained model and save them into csv files.

```
python3 test.py --model=logs/your_training_folder/model.pt --state_dict=logs/your_training_folder/state_dict.pt
```