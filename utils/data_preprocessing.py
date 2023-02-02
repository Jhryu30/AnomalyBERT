"""Parts of codes are brought from https://github.com/NetManAIOps/OmniAnomaly"""

import ast
import csv
import os
import sys
from pickle import dump
import json

import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def load_as_np(category, filename, dataset, dataset_folder, output_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float32,
                         delimiter=',')
    return temp


def load_data(dataset, base_dir, output_folder, json_folder):
    if dataset == 'SMD':
        dataset_folder = os.path.join(base_dir, 'OmniAnomaly/ServerMachineDataset')
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        
        train_files = []
        test_files = []
        label_files = []
        file_length = [0]
        for filename in file_list:
            if filename.endswith('.txt'):
                train_files.append(load_as_np('train', filename, filename.strip('.txt'), dataset_folder, output_folder))
                test_files.append(load_as_np('test', filename, filename.strip('.txt'), dataset_folder, output_folder))
                label_files.append(load_as_np('test_label', filename, filename.strip('.txt'), dataset_folder, output_folder))
                file_length.append(len(label_files[-1]))
        
        for i, train, test, label in zip(range(len(test_files)), train_files, test_files, label_files):
            np.save(os.path.join(output_folder, dataset + "{}_train.npy".format(i)), train)
            np.save(os.path.join(output_folder, dataset + "{}_test.npy".format(i)), test)
            np.save(os.path.join(output_folder, dataset + "{}_test_label.npy".format(i)), label)
            
        train_files = np.concatenate(train_files, axis=0)
        test_files = np.concatenate(test_files, axis=0)
        label_files = np.concatenate(label_files, axis=0)
        np.save(os.path.join(output_folder, dataset + "_train.npy"), train_files)
        np.save(os.path.join(output_folder, dataset + "_test.npy"), test_files)
        np.save(os.path.join(output_folder, dataset + "_test_label.npy"), label_files)
        
        file_length = np.cumsum(np.array(file_length)).tolist()
        channel_divisions = []
        for i in range(len(file_length)-1):
            channel_divisions.append([file_length[i], file_length[i+1]])
        with open(os.path.join(json_folder, dataset + "_" + 'test_channel.json'), 'w') as file:
            json.dump(channel_divisions, file)
                
    elif dataset == 'SMAP' or dataset == 'MSL':
        dataset_folder = os.path.join(base_dir, 'telemanom/data')
        with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0][0]+'-{:2d}'.format(int(k[0][2:])))        
#         label_folder = os.path.join(dataset_folder, 'test_label')
#         if not os.path.exists(label_folder):
#             os.mkdir(label_folder)
#         makedirs(label_folder, exist_ok=True)
        data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']
#         data_info = [row for row in res if row[1] == dataset]
        labels = []
        class_divisions = {}
        channel_divisions = []
        current_index = 0
    
        for row in data_info:
            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
            
            _class = row[0][0]
            if _class in class_divisions.keys():
                class_divisions[_class][1] += length
            else:
                class_divisions[_class] = [current_index, current_index+length]
            channel_divisions.append([current_index, current_index+length])
            current_index += length
            
        labels = np.asarray(labels)
#         print(dataset, 'test_label', labels.shape)
#         with open(os.path.join(output_folder, dataset + "_" + 'test_label' + ".pkl"), "wb") as file:
#             dump(labels, file)
        np.save(os.path.join(output_folder, dataset + "_" + 'test_label' + ".npy"), labels)
        
        with open(os.path.join(json_folder, dataset + "_" + 'test_class.json'), 'w') as file:
            json.dump(class_divisions, file)
        with open(os.path.join(json_folder, dataset + "_" + 'test_channel.json'), 'w') as file:
            json.dump(channel_divisions, file)

        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(os.path.join(dataset_folder, category, filename + '.npy'))
                data.extend(temp)
            data = np.asarray(data)
#             print(dataset, category, data.shape)
#             with open(os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
#                 dump(data, file)
            data = MinMaxScaler().fit_transform(data)
            np.save(os.path.join(output_folder, dataset + "_" + category + ".npy"), data)

        for c in ['train', 'test']:
            concatenate_and_save(c)
            
    elif dataset == 'SWaT':
        dataset_folder = os.path.join(base_dir, 'SWaT/Physical')
        normal_data = pd.read_excel(os.path.join(dataset_folder, 'SWaT_Dataset_Normal_v1.xlsx'))
        normal_data = normal_data.iloc[1:, 1:-1].to_numpy()
        normal_data = MinMaxScaler().fit_transform(normal_data).clip(0, 1)
        np.save(os.path.join(output_folder, dataset + "_train.npy"), normal_data)
        
        abnormal_data = pd.read_excel(os.path.join(dataset_folder, 'SWaT_Dataset_Attack_v0.xlsx'))
        abnormal_label = abnormal_data.iloc[1:, -1] == 'Attack'
        abnormal_label = abnormal_label.to_numpy().astype(int)
        
        abnormal_data = abnormal_data.iloc[1:, 1:-1].to_numpy()
        abnormal_data = MinMaxScaler().fit_transform(abnormal_data).clip(0, 1)
        np.save(os.path.join(output_folder, dataset + "_test.npy"), abnormal_data)
        np.save(os.path.join(output_folder, dataset + "_test_label.npy"), abnormal_label)
        
    elif dataset == 'WADI':
        normal_data = pd.read_csv(os.path.join(base_dir, 'WADI/WADI.A2_19 Nov 2019/WADI_14days_new.csv'))
        normal_data = normal_data.dropna(axis='columns', how='all').dropna()
        normal_data = normal_data.iloc[:, 3:].to_numpy()
        normal_data = MinMaxScaler().fit_transform(normal_data).clip(0, 1)
        np.save(os.path.join(output_folder, dataset + "_train.npy"), normal_data)
        
        abnormal_data = pd.read_csv(os.path.join(base_dir, 'WADI/WADI.A2_19 Nov 2019/WADI_attackdataLABLE.csv'), header=1)
        abnormal_data = abnormal_data.dropna(axis='columns', how='all').dropna()
        abnormal_label = abnormal_data.iloc[:, -1] == -1
        abnormal_label = abnormal_label.to_numpy().astype(int)
        
        abnormal_data = abnormal_data.iloc[:, 3:-1].to_numpy()
        abnormal_data = MinMaxScaler().fit_transform(abnormal_data).clip(0, 1)
        np.save(os.path.join(output_folder, dataset + "_test.npy"), abnormal_data)
        np.save(os.path.join(output_folder, dataset + "_test_label.npy"), abnormal_label)
        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str,
                        help="Name of dataset; SMD/SMAP/MSL/SWaT/WADI")
    parser.add_argument("--data_dir", required=True, type=str,
                        help="Directory of raw data")
    parser.add_argument("--out_dir", default=None, type=str,
                        help="Directory of the processed data")
    parser.add_argument("--json_dir", default=None, type=str,
                        help="Directory of the json files for the processed data")
    options = parser.parse_args()
    
    datasets = ['SMD', 'SMAP', 'MSL', 'SWaT', 'WADI']
    
    if options.dataset in datasets:
        base_dir = options.data_dir
        
        if options.out_dir == None:
            output_folder = os.path.join(base_dir, 'processed')
        else:
            output_folder = options.out_dir
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
            
        if options.json_dir == None:
            json_folder = os.path.join(base_dir, 'json')
        else:
            json_folder = options.json_dir
        if not os.path.exists(json_folder):
            os.mkdir(json_folder)
            
        load_data(options.dataset, base_dir, output_folder, json_folder)
    
#     commands = sys.argv[1:]
#     load = []
#     if len(commands) > 0:
#         for d in commands:
#             if d in datasets:
#                 load_data(d)
#     else:
#         print("""
#         Usage: python data_preprocess.py <datasets>
#         where <datasets> should be one of ['SMD', 'SMAP', 'MSL']
#         """)