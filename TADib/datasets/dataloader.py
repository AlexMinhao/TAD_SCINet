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
    "WADI": "./data/WADI",
    "SWAT_OLD": "./data/SWAT/processed",
    "SWAT": "./data/SWAT/processed",
    "WADI_SPLIT": "./datasets/anomaly/WADI_SPLIT/processed",
    "SWAT_SPLIT": "./datasets/anomaly/SWAT_SPLIT/processed",
    "gesture":"./data/gesture/labeled",
    "power_demand":"./data/power_demand/labeled",
    "nyc_taxi":"./data/nyc_taxi/labeled",
    "kdd99":"./data/kdd99",
    "ecg":"./data/ecg",
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
        return 40
    elif "gesture" in dataset:
        return 2
    elif "power" in dataset:
        return 1
    elif "nyc_taxi" in dataset:
        return 3
    elif "kdd99" in dataset:
        return 40
    elif "ecg" in dataset:
        return 2
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

    elif dataset == 'SWAT_OLD'or dataset =='SMD':
        prefix = subdataset
        downsample = False
        if dataset == 'SWAT_OLD':
            downsample = True
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
            if downsample:
                length = train_data.shape[0]
                index = np.arange(0,length,10)
                train_data = train_data[index]
            if do_preprocess:
                train_data = preprocess(train_data)
            if use_dim != "all":
                train_data = train_data[:, use_dim].reshape(-1, 1)
            if len(train_data) > 0:
                train_data_list.append(train_data)
        data_dict["train"] = np.concatenate(train_data_list, axis=0)[:nrows]
        dtrain = np.concatenate(train_data_list, axis=0)[:nrows]
        # np.savetxt(
        #     f'./data/SWAT/swat_train_process.csv',
        #     dtrain, delimiter=",")


        test_data_list = []
        for idx, f_name in enumerate(test_files):
            machine_name = os.path.basename(f_name).split("_")[0]
            f = open(f_name, "rb")
            test_data = pickle.load(f).reshape((-1, x_dim))
            f.close()
            if downsample:
                length = test_data.shape[0]
                index = np.arange(0,length,10)
                test_data = test_data[index]
            if do_preprocess:

                test_data = preprocess(test_data)
            if use_dim != "all":
                test_data = test_data[:, use_dim].reshape(-1, 1)
            if len(test_data) > 0:
                test_data_list.append(test_data)
        data_dict["test"] = np.concatenate(test_data_list, axis=0)[:nrows]

        label_data_list = []
        for idx, f_name in enumerate(label_files):
            machine_name = os.path.basename(f_name).split("_")[0]
            f = open(f_name, "rb")
            label_data = pickle.load(f)
            f.close()
            if downsample:
                length = label_data.shape[0]
                index = np.arange(0, length, 10)
                label_data = label_data[index]
            if len(label_data) > 0:
                # label_data[0:1000] = 0
                # label_data[44000:-1] = 0
                label_data_list.append(label_data)
        data_dict["test_labels"] = np.concatenate(label_data_list, axis=0)[:nrows]
        do_preprocess = True


        for k, v in data_dict.items():
            if k == "dim":
                continue
            print("Shape of {} is {}.".format(k, v.shape))
        return data_dict
    elif dataset == 'SWAT' or dataset == 'WADI':
        if dataset == 'SWAT':
            data_dict = load_swat_data(path)
            return data_dict

        else:
            train_df, val_df, test_df = load_wadi_data(path)
            return train_df


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


def load_wadi_data(path):
    train = path+'/WADI_train.zip'
    z_tr = zipfile.ZipFile(train, "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    train_df = pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()

    test = path + '/WADI_test.zip'
    z_tr = zipfile.ZipFile(test, "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    test_df = pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()

    train_df = train_df.fillna(method='ffill')
    test_df.loc[test_df['label'] >= 1, 'label'] = 1
    test_df = test_df.fillna(method='ffill')

    sensors = ['1_AIT_001_PV', '1_AIT_002_PV', '1_AIT_003_PV', '1_AIT_004_PV',
               '1_AIT_005_PV', '1_FIT_001_PV', '1_LT_001_PV', '2_DPIT_001_PV',
               '2_FIC_101_CO', '2_FIC_101_PV', '2_FIC_101_SP', '2_FIC_201_CO',
               '2_FIC_201_PV', '2_FIC_201_SP', '2_FIC_301_CO', '2_FIC_301_PV',
               '2_FIC_301_SP', '2_FIC_401_CO', '2_FIC_401_PV', '2_FIC_401_SP',
               '2_FIC_501_CO', '2_FIC_501_PV', '2_FIC_501_SP', '2_FIC_601_CO',
               '2_FIC_601_PV', '2_FIC_601_SP', '2_FIT_001_PV', '2_FIT_002_PV',
               '2_FIT_003_PV', '2_FQ_101_PV', '2_FQ_201_PV', '2_FQ_301_PV', '2_FQ_401_PV',
               '2_FQ_501_PV', '2_FQ_601_PV', '2_LT_001_PV', '2_LT_002_PV', '2_MCV_101_CO',
               '2_MCV_201_CO', '2_MCV_301_CO', '2_MCV_401_CO', '2_MCV_501_CO', '2_MCV_601_CO',
               '2_P_003_SPEED', '2_P_004_SPEED', '2_PIC_003_CO', '2_PIC_003_PV', '2_PIT_001_PV',
               '2_PIT_002_PV', '2_PIT_003_PV', '2A_AIT_001_PV', '2A_AIT_002_PV', '2A_AIT_003_PV',
               '2A_AIT_004_PV', '2B_AIT_001_PV', '2B_AIT_002_PV', '2B_AIT_003_PV', '2B_AIT_004_PV',
               '3_AIT_001_PV', '3_AIT_002_PV', '3_AIT_003_PV', '3_AIT_004_PV', '3_AIT_005_PV',
               '3_FIT_001_PV', '3_LT_001_PV', 'LEAK_DIFF_PRESSURE', 'TOTAL_CONS_REQUIRED_FLOW']

    actuators = ['1_MV_001_STATUS', '1_MV_004_STATUS', '1_P_001_STATUS', '1_P_003_STATUS',
                 '1_P_005_STATUS', '2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH', '2_LS_201_AL',
                 '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH',
                 '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL', '2_MV_003_STATUS', '2_MV_006_STATUS',
                 '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS',
                 '2_MV_501_STATUS', '2_MV_601_STATUS', '2_P_003_STATUS']

    # signals = []
    # for name in sensors:
    #     signals.append(ContinousSignal(name, SignalSource.sensor, isInput=True, isOutput=True,
    #                                    min_value=train_df[name].min(), max_value=train_df[name].max(),
    #                                    mean_value=train_df[name].mean(), std_value=train_df[name].std()))
    # for name in actuators:
    #     signals.append(DiscreteSignal(name, SignalSource.controller, isInput=True, isOutput=False,
    #                                   values=train_df[name].unique()))

    pos = len(train_df) * 3 // 4
    val_df = train_df.loc[pos:, :]
    val_df = val_df.reset_index(drop=True)

    train_df = train_df.loc[:pos, :]
    train_df = train_df.reset_index(drop=True)
    return train_df, val_df, test_df#, signals


def load_swat_data(path):
    train = path + '/SWat_train.zip'
    z_tr = zipfile.ZipFile(train, "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    train_df = pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()

    test = path + '/SWat_test.zip'
    z_tr = zipfile.ZipFile(test, "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    test_df = pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()

    test_df['label'] = 0
    test_df.loc[test_df['Normal/Attack'] != 'Normal', 'label'] = 1



    sensors = ['FIT101', 'LIT101', 'AIT201', 'AIT202', 'AIT203', 'FIT201',
               'DPIT301', 'FIT301', 'LIT301', 'AIT401', 'AIT402', 'FIT401',
               'LIT401', 'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501',
               'FIT502', 'FIT503', 'FIT504', 'PIT501', 'PIT502', 'PIT503', 'FIT601', ]

    actuators = ['MV101', 'P101', 'P102', 'MV201', 'P201', 'P202',
                 'P203', 'P204', 'P205', 'P206', 'MV301', 'MV302',
                 'MV303', 'MV304', 'P301', 'P302', 'P401', 'P402',
                 'P403', 'P404', 'UV401', 'P501', 'P502', 'P601',
                 'P602', 'P603']
    # signals = []
    # for name in sensors:
    #     signals.append(ContinousSignal(name, SignalSource.sensor, isInput=True, isOutput=True,
    #                                    min_value=train_df[name].min(), max_value=train_df[name].max(),
    #                                    mean_value=train_df[name].mean(), std_value=train_df[name].std()))
    # for name in actuators:
    #     signals.append(DiscreteSignal(name, SignalSource.controller, isInput=True, isOutput=False,
    #                                   values=train_df[name].unique()))
    length = train_df.shape[0]
    index = np.arange(0, length, 2)
    train_data = train_df.values[:,1:52]
    train_df_sampled =train_data[index][16530:,:]
    # pos = len(train_df) * 3 // 4
    # val_df = train_data[pos:, :]
    # val_df = val_df.reset_index(drop=True)

    # train_df = train_df.loc[:pos, :]
    # train_df = train_df.reset_index(drop=True)
    length = test_df.shape[0]
    index = np.arange(0, length, 2)
    test_data = test_df.values[:,1:52]
    test_label = test_df.values[:,-1]
    test_df_sampled = test_data[index]
    test_df_sampled_label = test_label[index]

    data_dict = defaultdict(dict)

    data_max = np.max(train_df_sampled, axis=0)
    data_min = np.min(train_df_sampled, axis=0)
    # train_normalize_statistic = {"max": data_max.tolist(), "min": data_min.tolist()}

    scale = data_max - data_min + 1e-5
    train_df_sampled = (train_df_sampled - data_min) / scale
    index = np.where(scale != 1e-5)
    index_not = [4, 10, 11, 13, 15, 23, 29, 30, 31, 32, 33 ,42, 43, 48 ,50]
    index = [ 0,  1,  2,  3,  5,  6 , 7 , 8 , 9 ,12 ,14 ,16, 17 ,18 ,19, 20 ,21 ,22, 24, 25, 26, 27, 28, 34,
 35, 36 ,37 ,38, 39, 40 ,41 ,44, 45 ,46, 47, 49]

    # data = np.clip(data, 0.0, 1.0)

    train_df_sampled = np.array(train_df_sampled[:,index],dtype=float)
    test_df_sampled = np.array(test_df_sampled[:,index],dtype=float)
    test_df_sampled_label = np.array(test_df_sampled_label,dtype=float)
    data_min = data_min[index]
    scale = scale[index]

    test_df_sampled = (test_df_sampled - data_min) / scale
    test_df_sampled = np.array(test_df_sampled, dtype=float)

    np.savetxt(
        f'./data/SWAT/swat_train_process.csv',
        train_df_sampled, delimiter=",")
    np.savetxt(
        f'./data/SWAT/swat_test_process.csv',
        test_df_sampled, delimiter=",")
    np.savetxt(
        f'./data/SWAT/swat_test_process_label.csv',
        test_df_sampled_label, delimiter=",")

    data_dict["train"] = train_df_sampled
    data_dict["test"] = test_df_sampled
    data_dict["test_labels"] = test_df_sampled_label
    data_dict["dim"] = 51

    return data_dict #, signals





if __name__ == "__main__":

    load_dataset('WADI', subdataset = 'swat', use_dim="all", root_dir="/", nrows=None)