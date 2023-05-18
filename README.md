# AnomalyBERT: Transformer-based Anomaly Detector

This is the code for **Self-supervised Transformer for Time Series Anomaly Detection using Data Degradation Scheme**.

## Installation

Please clone our repository at `path/to/repository/` and install the packages in `requirements.txt`.
Before installing the packages, we recommend installing Python 3.8 and Pytorch 1.9 with CUDA.

```
git clone https://github.com/Jhryu30/AnomalyBERT.git path/to/repository/

conda create --name your_env_name python=3.8
conda activate your_env_name

pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html  # example CUDA setting
pip install -r requirements.txt
```

We use five public datasets, SMAP, MSL, SMD, SWaT, and WADI.
Following the instruction in [here](utils/DATA_PREPARATION.md), you can download and preprocess the datasets.
After preprocessing, you need to edit your dataset directory in `utils/config.py`.

```
DATASET_DIR = 'path/to/dataset/processed/'
```

## Demo

We release our trained models on SWaT/WADI/SMAP/MSL. 
You can download the files from [here](https://drive.google.com/drive/folders/1PhMwdGsSnrQgs16DPgBPngwV6Fvliatd?usp=sharing), and we recommend placing it in `logs/best_checkpoints/` folder.
Now you can run our demo code in `demo.ipynb` and see how AnomalyBERT works.


## Training

We provide the training code for our model.
For example, to train a model of 6-layer Transformer body on SMAP dataset, run:

```
python3 train.py --dataset=SMAP --n_layer=6
```

To train a model on MSL dataset with patch size of 2 and customized outlier synthesis probability, run:

```
python3 train.py --dataset=MSL --patch_size=2 --soft_replacing=0.5 --uniform_replacing=0.1 --peak_noising=0.1 \
--length_adjusting=0.1
```

You can use the default option for training each dataset, as we did in our paper.

```
python3 train.py --default_options=SMAP # or any dataset name in MSL/SMD/SWaT/WADI and subset of SMD; SMD0 ~ SMD27
```

If you want to customize the model and training settings, please check the options in `train.py`.

## Anomaly score estimation and metric computation

To estimate anomaly scores of test data with the trained model, run the `estimate.py` code.
For example, you can estimate anomaly scores of SMAP test set divided by channel with window sliding of 16.

```
python3 estimate.py --dataset=SMAP --model=logs/YYMMDDhhmmss_SMAP/model.pt --state_dict=logs/YYMMDDhhmmss_SMAP/state_dict.pt \
--window_sliding=16
```

Now you will obtain results (npy) file that contains the estimated anomaly scores.
With the results file, you can compute F1-score with and without the point adjustment by running:

```
python3 compute_metrics.py --dataset=SMAP --result=logs/YYMMDDhhmmss_SMAP/state_dict_results.npy
```

If you want to customize the estimation or computation settings, please check the options in `estimate.py` and `compute_metrics.py`.