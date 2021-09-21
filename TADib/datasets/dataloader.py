import logging
import os
import pickle
from collections import defaultdict
from glob import glob
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import pandas as pd
import zipfile


data_path_dict = {
    "SMD": "./data/SMD",
    "SMAP": "./datasets/anomaly/SMAP-MSL/processed_SMAP",
    "MSL": "./data/MSL",
    "SWAT": "./data/SWAT",
    "gesture":"./data/gesture/labeled",
    "power_demand":"./data/power_demand/labeled",
    "nyc_taxi":"./data/nyc_taxi/labeled",
    "kdd99":"./data/kdd99",
    "ecg":"./data/ecg",
    "credit":"./data/credit",
}


def get_data_dim(dataset):
    if "SMAP" in dataset:
        return 25
    elif "MSL" in dataset:
        return 55 # 27
    elif "SMD" in dataset:
        return 38
    elif "WADI" in dataset:
        return 93
    elif "SWAT" in dataset:
        return 36
    elif "gesture" in dataset:
        return 2
    elif "power_demand" in dataset:
        return 1
    elif "nyc_taxi" in dataset:
        return 3
    elif "kdd99" in dataset:
        return 40
    elif "ecg" in dataset:
        return 2
    elif "credit" in dataset:
        return 28
    else:
        raise ValueError("unknown dataset " + str(dataset))


def load_dataset(dataset, subdataset, use_dim="all", root_dir="../", nrows=None):
    """
    use_dim: dimension used in multivariate timeseries
    """
    logging.info("Loading {} of {} dataset".format(subdataset, dataset))
    x_dim = get_data_dim(dataset)
    path = data_path_dict[dataset]

    if dataset == 'MSL' or dataset =='SMAP' :
        train_start = 0
        test_start = 0
        train_end = None
        test_end = None
        print('load data of:', dataset)
        print("train: ", train_start, train_end)  # train:  0 None
        print("test: ", test_start, test_end)  # test:  0 None

        f = open(os.path.join('./data/', dataset, dataset + '_train.pkl'), "rb")
        train_data = pickle.load(f)  # (28479, 38)
        train_data = train_data.reshape((-1, x_dim))[train_start:train_end, :]
        f.close()
        try:
            f = open(os.path.join('./data/', dataset, dataset + '_test.pkl'), "rb")
            test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]  # (28479, 38)
            f.close()
        except (KeyError, FileNotFoundError):
            test_data = None
        try:
            f = open(os.path.join('./data/', dataset, dataset + "_test_label.pkl"), "rb")
            test_label = pickle.load(f).reshape((-1))[test_start:test_end]  # 28479
            f.close()
        except (KeyError, FileNotFoundError):
            test_label = None
        do_preprocess = True
        if do_preprocess:
            train_data = preprocess(train_data)
            test_data = preprocess(test_data)
        print("train set shape: ", train_data.shape)
        print("test set shape: ", test_data.shape)
        print("test set label shape: ", test_label.shape)

        data_dict = defaultdict(dict)
        data_dict["train"] = train_data
        data_dict["test"] = test_data
        data_dict["test_labels"] = test_label
        data_dict["dim"] = x_dim

        return data_dict

    elif dataset == 'cover' or dataset =='pima' or dataset =='smtp' :

        train_start = 0
        test_start = 0
        train_end = None
        test_end = None
        print('load data of:', dataset)
        print("train: ", train_start, train_end)  # train:  0 None
        print("test: ", test_start, test_end)  # test:  0 None

        f = os.path.join('./data/', dataset+'.csv')
        data = np.loadtxt(f, dtype= np.float, delimiter=',')  # (28479, 38)
        length = data.shape[0]
        train_end = int(0.5*length)
        train_data = data[train_start:train_end, 0:-1]
        test_data = data[train_end::, 0:-1]
        test_data_label = data[train_end::, -1]

        if dataset =='smtp' or dataset == 'cover':
            length = train_data.shape[0]
            index = np.arange(0, length, 10)
            train_data = train_data[index]

            length = test_data.shape[0]
            index = np.arange(0, length, 10)
            test_data = test_data[index]

            test_data_label = test_data_label[index]

        data_max = np.max(train_data, axis=0)
        data_min = np.min(train_data, axis=0)
        # train_normalize_statistic = {"max": data_max.tolist(), "min": data_min.tolist()}

        scale = data_max - data_min + 1e-5
        train_data_scale = (train_data - data_min) / scale

        test_data_scale = (test_data - data_min) / scale
        data_dict = defaultdict(dict)
        data_dict["train"] = train_data_scale
        data_dict["test"] = test_data_scale
        data_dict["test_labels"] = test_data_label


        return data_dict

    elif dataset == 'credit':

        train_start = 0
        test_start = 0
        train_end = None
        test_end = None
        print('load data of:', dataset)
        print("train: ", train_start, train_end)  # train:  0 None
        print("test: ", test_start, test_end)  # test:  0 None

        f = os.path.join('./data/', dataset, 'Credit.csv')
        data = np.loadtxt(f, dtype= np.float, delimiter=',')  # (28479, 38)
        length = data.shape[0]
        train_end = int(0.5*length)
        train_data = data[train_start:train_end, 0:3]
        test_data = data[train_end::, 0:3]
        test_data_label = data[train_end::, 29]

        # length = train_data.shape[0]
        # index = np.arange(0, length, 10)
        # train_data = train_data[index]
        #
        # length = test_data.shape[0]
        # index = np.arange(0, length, 10)
        # test_data = test_data[index]
        #
        # test_data_label = test_data_label[index]

        data_max = np.max(train_data, axis=0)
        data_min = np.min(train_data, axis=0)
        # train_normalize_statistic = {"max": data_max.tolist(), "min": data_min.tolist()}

        scale = data_max - data_min + 1e-5
        train_data_scale = (train_data - data_min) / scale

        test_data_scale = (test_data - data_min) / scale
        data_dict = defaultdict(dict)
        data_dict["train"] = train_data_scale
        data_dict["test"] = test_data_scale
        data_dict["test_labels"] = test_data_label
        data_dict["dim"] = x_dim

        return data_dict


    elif dataset == 'SWAT' or dataset =='SMD':

        prefix ='swat'
        train_files = glob(os.path.join(path, prefix + "_train.pkl"))
        test_files = glob(os.path.join(path, prefix + "_test.pkl"))
        label_files = glob(os.path.join(path, prefix + "_test_label.pkl"))
        logging.info("{} files found.".format(len(train_files)))

        data_dict = defaultdict(dict)
        data_dict["dim"] = x_dim if use_dim == "all" else 1
        do_preprocess = True
        train_data_list = []
        for idx, f_name in enumerate(train_files):
            machine_name = os.path.basename(f_name).split("_")[0]
            f = open(f_name, "rb")
            train_data = pickle.load(f).reshape((-1, x_dim))
            f.close()
            if len(train_data) > 0:
                train_data_list.append(train_data)
        data_dict["train"] = np.concatenate(train_data_list, axis=0)[:nrows]
        dtrain = np.concatenate(train_data_list, axis=0)[:nrows]



        test_data_list = []
        for idx, f_name in enumerate(test_files):
            machine_name = os.path.basename(f_name).split("_")[0]
            f = open(f_name, "rb")
            test_data = pickle.load(f).reshape((-1, x_dim))
            f.close()
            if len(test_data) > 0:
                test_data_list.append(test_data)
        data_dict["test"] = np.concatenate(test_data_list, axis=0)[:nrows]

        label_data_list = []
        for idx, f_name in enumerate(label_files):
            machine_name = os.path.basename(f_name).split("_")[0]
            f = open(f_name, "rb")
            label_data = pickle.load(f)
            f.close()
            if len(label_data) > 0:
                label_data_list.append(label_data)
        data_dict["test_labels"] = np.concatenate(label_data_list, axis=0)[:nrows]

        for k, v in data_dict.items():
            if k == "dim":
                continue
            print("Shape of {} is {}.".format(k, v.shape))
        return data_dict




    elif dataset == 'ecg' or dataset == 'gesture' or dataset == 'nyc_taxi' or dataset == 'power_demand':
        data_dict = defaultdict(dict)
        data_dict["dim"] = x_dim
        if dataset == 'ecg':
            path_train = os.path.join('./data/', dataset, 'labeled/train/', subdataset + '.pkl')
        else:
            path_train = os.path.join('./data/', dataset, 'labeled/train/', dataset + '.pkl')
        with open(str(path_train), 'rb') as f_train:
            train = pickle.load(f_train)
            train = torch.FloatTensor(train)
            train_label = train[:, -1]
            train_data = train[:, :-1]
            mean = train_data.mean(dim=0)
            std = train_data.std(dim=0)
            length = len(train_data)
            # data, label = augmentation(data, label)
            train_data = standardization(train_data, mean, std)
        if dataset == 'ecg':
            path_test = os.path.join('./data/', dataset, 'labeled/test/', subdataset + '.pkl')
        else:
            path_test = os.path.join('./data/', dataset, 'labeled/test/', dataset + '.pkl')
        with open(str(path_test), 'rb') as f_test:
            test = pickle.load(f_test)
            test = torch.FloatTensor(test)
            test_label = test[:, -1]
            test_data = test[:, :-1]
            length = len(test_data)
            # data, label = augmentation(data, label)


            test_data = standardization(test_data, mean, std)

        data_dict["test_labels"] = test_label
        data_dict["train"] = train_data
        data_dict["test"] = test_data
        return data_dict

    elif dataset == 'kdd99':
        f = os.path.join('./data/', dataset, dataset + '_train.npy')
        train_data = np.load(f)  # (28479, 38)
        m, n = train_data.shape  # m=562387, n=35
        downsample = True
        # normalization
        for i in range(n - 1):
            # print('i=', i)
            A = max(train_data[:, i])
            # print('A=', A)
            if A != 0:
                train_data[:, i] /= max(train_data[:, i])
                # scale from -1 to 1
                train_data[:, i] = 2 * train_data[:, i] - 1
            else:
                train_data[:, i] = train_data[:, i]

        train = train_data[:, 0:n - 1]
        train_labels = train_data[:, n - 1]
        if downsample:
            length = train.shape[0]
            index = np.arange(0, length, 10)
            train = train[index]


        f = os.path.join('./data/', dataset, dataset + '_test.npy')
        test_data = np.load(f)  # (28479, 38)
        m, n = test_data.shape  # m1=494021, n1=35

        for i in range(n - 1):
            B = max(test_data[:, i])
            if B != 0:
                test_data[:, i] /= max(test_data[:, i])
                # scale from -1 to 1
                test_data[:, i] = 2 * test_data[:, i] - 1
            else:
                test_data[:, i] = test_data[:, i]

        test = test_data[:, 0:n - 1]
        test_labels = test_data[:, n - 1]
        if downsample:
            length = test.shape[0]
            index = np.arange(0, length, 10)
            test = test[index]

        data_dict = defaultdict(dict)
        data_dict["dim"] = n
        data_dict["test_labels"] = test_labels
        data_dict["train"] = train
        data_dict["test"] = test
        return data_dict

    else:
        print('No datasets')

    # if subdataset:
    #     prefix = subdataset
    #     train_files = glob(os.path.join(path, prefix + "_train.pkl"))
    #     test_files = glob(os.path.join(path, prefix + "_test.pkl"))
    #     label_files = glob(os.path.join(path, prefix + "_test_label.pkl"))
    # else:
    #
    #     train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0) #1565 27
    #     test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)#2049 28
    #     feature_file = open(f'./data/{dataset}/list.txt', 'r')
    #     feature_list = []
    #     for ft in feature_file:
    #         feature_list.append(ft.strip())
    #
    #
    #     train, test = train_orig, test_orig
    #     train_dataset_indata = construct_data(train, feature_list, labels=0)  # 1565 27
    #     test_dataset_indata = construct_data(test, feature_list, labels=test.attack.tolist())  # 2049 28
    #     return train_dataset_indata, test_dataset_indata


def augmentation(self,data,label,noise_ratio=0.05,noise_interval=0.0005,max_length=100000):
    noiseSeq = torch.randn(data.size())
    augmentedData = data.clone()
    augmentedLabel = label.clone()
    for i in np.arange(0, noise_ratio, noise_interval):
        scaled_noiseSeq = noise_ratio * self.std.expand_as(data) * noiseSeq
        augmentedData = torch.cat([augmentedData, data + scaled_noiseSeq], dim=0)
        augmentedLabel = torch.cat([augmentedLabel, label])
        if len(augmentedData) > max_length:
            augmentedData = augmentedData[:max_length]
            augmentedLabel = augmentedLabel[:max_length]
            break

    return augmentedData, augmentedLabel


def construct_data(data, feature_map, labels=0):
    res = []

    for feature in feature_map:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, 'not exist in data')
    # append labels as last
    sample_n = len(res[0])

    if type(labels) == int:
        res.append([labels]*sample_n)
    elif len(labels) == sample_n:
        res.append(labels)

    return res

def preprocess(df):
    """returns normalized and standardized data.
    """

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()

    # normalize data
    df = MinMaxScaler().fit_transform(df)
    print('Data normalized')

    return df


def normalization(seqData,max,min):
    return (seqData -min)/(max-min)

def standardization(seqData,mean,std):
    return (seqData-mean)/std

def reconstruct(seqData,mean,std):
    return seqData*std+mean








if __name__ == "__main__":

    load_dataset('power_demand', subdataset = '0', use_dim="all", root_dir="/", nrows=None)