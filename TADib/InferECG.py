import pandas as pd
import numpy as np
import torch
from torch import nn

from datasets.dataloader import load_dataset
import os
from datasets.SNetDataset import SNetDataset

import time
import torch.nn.functional as F


from models.SCINetPWPretrainMask import SCI_Point_Mask
from models.SCINetBiEvenSeqPretrain import SCIMaskEvenPretrain
from sklearn.metrics import f1_score, mean_squared_error
from sklearn import metrics
from sklearn.metrics import auc

from tslearn.metrics import dtw
import matplotlib.pyplot as plt

from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores
from torch.utils.data import DataLoader, random_split, Subset
import random
import argparse
from datetime import datetime
from pathlib import Path
import math

parser = argparse.ArgumentParser()

parser.add_argument('--batch', help='batch size', type = int, default=16)
parser.add_argument('--epoch', help='train epoch', type = int, default=30)
parser.add_argument('--learning_rate', help='lr', type = float, default=0.001)

parser.add_argument('--slide_win', help='slide_win', type = int, default=168)

parser.add_argument('--slide_stride', help='slide_stride', type = int, default=16)
parser.add_argument('--save_path_pattern', help='save path pattern', type = str, default='msl')
parser.add_argument('--dataset', help='wadi / ECG', type = str, default='')
parser.add_argument('--device', help='cuda / cpu', type = str, default='cuda')
parser.add_argument('--random_seed', help='random seed', type = int, default=4321)
parser.add_argument('--comment', help='experiment comment', type = str, default='')
parser.add_argument('--out_layer_num', help='outlayer num', type = int, default=1)
parser.add_argument('--out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
parser.add_argument('--decay', help='decay', type = float, default=0)
parser.add_argument('--val_ratio', help='val ratio', type = float, default=0.1)
parser.add_argument('--topk', help='topk num', type = int, default=20)
parser.add_argument('--report', help='best / val', type = str, default='best')
parser.add_argument('--load_model_path', help='trained model path', type = str, default='')

parser.add_argument('--hidden-size', default=3, type=float, help='hidden channel of module')
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
parser.add_argument('--kernel', default=3, type=int, help='kernel size')
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--save_path', type=str, default='ECG_Results')

parser.add_argument('--pred_len', type=int, default=128)
parser.add_argument('--seq_mask_range_low', type=int, default=8)
parser.add_argument('--seq_mask_range_high', type=int, default=4)

parser.add_argument('--ensemble', type=int, default=0)
parser.add_argument('--model_type', type=str, default='BiPointMask')
parser.add_argument('--model_name', type=str, default='SCINet')
parser.add_argument('--point_part', type=int, default=12)

parser.add_argument('--variate_index', type=int, default=1)

parser.add_argument('--lradj', type=int, default=3,help='adjust learning rate')

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."
    #
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."
    #
    # if window_len < 3:
    #     return x
    #
    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def test(model, dataloader, type = 0):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = args.device

    test_loss_list = []
    now = time.time()

    t_test_mask_list = []
    t_test_input_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels in dataloader:
        x, y, labels = [item.to(device).float() for item in [x, y, labels]]
        # torch.Size([31, 27, 15])  torch.Size([31, 27])
        with torch.no_grad():

            predicted = model(x)  # torch.Size([31, 27])


            if isinstance(predicted,tuple):

                predicted = predicted[0]
                loss = loss_func(predicted, y) #+ loss_func(predicted[1], y)
            else:
                predicted = predicted.float().to(device)
                loss = loss_func(predicted, y)

            # labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])  # torch.Size([31, 27])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels


            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)


        test_loss_list.append(loss.item())
        acu_loss += loss.item()

        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    # t_test_predicted_list=t_test_predicted_list.permute(0,2,1)
    data_dim = t_test_predicted_list.shape[2]


    t_test_predicted_list = t_test_predicted_list.reshape(-1,data_dim)

    # t_test_ground_list = t_test_ground_list.permute(0, 2, 1)
    t_test_ground_list = t_test_ground_list.reshape(-1, data_dim)


    t_test_labels_list = t_test_labels_list.reshape(-1, 1)  #torch.Size([2048, 1])

    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()
    test_labels_list = t_test_labels_list.tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)




    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]





def get_score(test_result, val_result):

    feature_num = len(test_result[0][0])
    np_test_result = np.array(test_result)  # 3 2034 27
    np_val_result = np.array(val_result)  # 3*31*27

    test_labels = np_test_result[2, :].tolist()

    test_scores, normal_scores = get_full_err_scores(test_result, val_result)  # 27 2034  27 31

    top1_best_info = get_best_performance_data(test_scores, test_labels,
                                               topk=1)  # max(final_topk_fmeas), pre, rec, auc_score, thresold, pred_labels
    top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)

    print('=========================** Result **============================\n')

    info = None
    info = top1_best_info


    print(f'F1 score: {info[0]}')
    print(f'precision: {info[1]}')
    print(f'recall: {info[2]}\n')

def get_loaders(train_dataset, seed, batch, val_ratio=0.1):
    dataset_len = int(len(train_dataset))
    train_use_len = int(dataset_len * (1 - val_ratio))  #279
    val_use_len = int(dataset_len * val_ratio) #0.1
    val_start_index = random.randrange(train_use_len) #197
    indices = torch.arange(dataset_len)

    train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]]) #0-197  228-309
    train_subset = Subset(train_dataset, train_sub_indices)

    val_sub_indices = indices[val_start_index:val_start_index+val_use_len]  #197-228
    val_subset = Subset(train_dataset, val_sub_indices)


    train_dataloader = DataLoader(train_subset, batch_size=batch,
                            shuffle=True)

    val_dataloader = DataLoader(val_subset, batch_size=batch,
                            shuffle=False)

    return train_dataloader, val_dataloader


def infer(model, dataloader, type = 0):
    # test
    loss_func = nn.L1Loss()
    device = args.device

    test_loss_list = []
    now = time.time()

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels in dataloader:
        x, y, labels = [item.to(device).float() for item in [x, y, labels]]
        # torch.Size([31, 27, 15])  torch.Size([31, 27])
        with torch.no_grad():
            # x = x.permute(0, 2, 1)  # torch.Size([128, 16, 27])
            # y = y.permute(0, 2, 1)  # torch.Size([128, 16, 27])
            x_temp = x.detach().cpu().numpy()
            predicted = model(x)  # torch.Size([31, 27])



            predicted = predicted.float().to(device)
            loss = loss_func(predicted, y)

            # labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])  # torch.Size([31, 27])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels


            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)


        test_loss_list.append(loss.item())
        acu_loss += loss.item()

        i += 1

        # if i % 10000 == 1 and i > 1:
        #     print(timeSincePlus(now, i / test_len))

    # t_test_predicted_list=t_test_predicted_list.permute(0,2,1)
    data_dim = t_test_predicted_list.shape[2]


    t_test_predicted_list = t_test_predicted_list.reshape(-1,data_dim)

    # t_test_ground_list = t_test_ground_list.permute(0, 2, 1)
    t_test_ground_list = t_test_ground_list.reshape(-1, data_dim)


    t_test_labels_list = t_test_labels_list.reshape(-1, 1)  #torch.Size([2048, 1])
    # if type == 1:
    #     folder_path = './checkpoint/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #     print(folder_path)
    #     np.savetxt( #args.model_type
    #         f'{folder_path}/NewDataBiEvenSeqMaskECG/{args.variate_index}dim/val_{args.dataset}_{args.model_type}2Stack_{args.variate_index}dimNum{args.variate_index}_group{args.groups}_lradj{args.lradj}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_ground_list.csv',
    #         t_test_ground_list.detach().cpu().numpy(), delimiter=",")
    #     np.savetxt(
    #         f'{folder_path}/NewDataBiEvenSeqMaskECG/{args.variate_index}dim/val_{args.dataset}_{args.model_type}2Stack_{args.variate_index}dimNum{args.variate_index}_group{args.groups}_lradj{args.lradj}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_predicted_list.csv',
    #         t_test_predicted_list.detach().cpu().numpy(), delimiter=",")
    #     np.savetxt(
    #         f'{folder_path}/NewDataBiEvenSeqMaskECG/{args.variate_index}dim/val_{args.dataset}_{args.model_type}2Stack_{args.variate_index}dimNum{args.variate_index}_group{args.groups}_lradj{args.lradj}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_labels_list.csv',
    #         t_test_labels_list.detach().cpu().numpy(), delimiter=",")


    # elif type == 2:
    #     folder_path = './checkpoint/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #     print(folder_path)
    #     np.savetxt(
    #         f'{folder_path}/NewDataBiEvenSeqMaskECG/{args.variate_index}dim/test_{args.dataset}_{args.model_type}2Stack_{args.variate_index}dimNum{args.variate_index}_group{args.groups}_lradj{args.lradj}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_ground_list.csv',
    #         t_test_ground_list.detach().cpu().numpy(), delimiter=",")
    #     np.savetxt(
    #         f'{folder_path}/NewDataBiEvenSeqMaskECG/{args.variate_index}dim/test_{args.dataset}_{args.model_type}2Stack_{args.variate_index}dimNum{args.variate_index}_group{args.groups}_lradj{args.lradj}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_predicted_list.csv',
    #         t_test_predicted_list.detach().cpu().numpy(), delimiter=",")
    #     np.savetxt(
    #         f'{folder_path}/NewDataBiEvenSeqMaskECG/{args.variate_index}dim/test_{args.dataset}_{args.model_type}2Stack_{args.variate_index}dimNum{args.variate_index}_group{args.groups}_lradj{args.lradj}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_labels_list.csv',
    #         t_test_labels_list.detach().cpu().numpy(), delimiter=",")




    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()
    test_labels_list = t_test_labels_list.tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    actual = np.asarray(actual)
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (1.5 * precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1

        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True

    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for k in range(i, len(score)-1, 1):
                    if not actual[k]:
                        break
                    else:
                        if not predict[k]:
                            predict[k] = True


        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True



    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict
def get_dtw(errors,labels, data_len, interval = 32):
    error = np.array(errors)
    label = np.array(labels)


    min_error = min(error)
    max_error = max(error)
    rang = range(128, data_len - interval, interval)
    fineture_range= np.arange(min_error, max_error,0.01)
    best_f1 = []
    Finetunelabel = []
    Threshold = []
    TPs =[]
    FPs =[]
    Pres = []
    Recs = []
    for K in fineture_range:
        finetunelabel = np.zeros(label.shape[0])
        sc = error>K
        for k in range(len(error)):
            if sc[k] > 0:
                for kk in range(rang[k],rang[k]+interval,1):
                    finetunelabel[kk] = 1

        f1, precision, recall, TP, TN, FP, FN = calc_point2point(finetunelabel, label)
        TPR = TP/  ( TP +  FN+ 0.00001)
        FPR = FP / ( FP + TN + 0.00001)
        TPs.append(TPR)
        FPs.append(FPR)
        Pres.append(precision)
        Recs.append(recall)
        best_f1.append(f1)
        Finetunelabel.append(finetunelabel)
        Threshold.append(K)

    best_f1_final = np.max(best_f1)
    best_f1_final_index = np.where(best_f1==np.max(best_f1))
    best_Threshold = Threshold[int(best_f1_final_index[0][0])]

    AUROC = metrics.auc(FPs,TPs)
    AUPRC = metrics.auc(Recs, Pres)
    print(f'Init F1 score: {best_f1_final},Threshods: {best_Threshold}')
    print(f'AUROC: {AUROC}, AUPRC: {AUPRC}')

    pred_labels = Finetunelabel[int(best_f1_final_index[0][0])]

    # max(final_topk_fmeas), pre, rec, auc_score, thresold, pred_labels
    predict = adjust_predicts(score=label, label=label,
                              threshold=0,
                              pred=pred_labels,
                              calc_latency=False)

    f1, precision, recall, TP, TN, FP, FN = calc_point2point(predict, label)
    print(f'Seq F1 score: {f1}')
    print(f'Seq precision: {precision}')
    print(f'Seq recall: {recall}\n')

    # plt.plot(rang, error)
    plt.plot(label*5,'.')
    plt.plot(pred_labels * 6,'.')
    plt.show()

if __name__ == "__main__":
    from sklearn import metrics
    from sklearn.metrics import auc
    import numpy as np
    y = np.array([1,1,2,2])
    sore = np.array([0.1,0.4,0.35,0.8])
    fpr,tpr,thred = metrics.roc_curve(y,sore,pos_label=2)


    torch.manual_seed(1234)  # reproducible
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True
    # 'machine-1-1'
    args.dataset = 'ecg'
    # chfdb_chf01_275_new  chfdb_chf13_45590   chfdbchf15   ltstdb_20221_43  ltstdb_20321_240   mitdb__100_180

    data_dict = load_dataset(args.dataset, subdataset='chfdb_chf01_275_new', use_dim="all", root_dir="/", nrows=None)


    args.variate_index = 2

    args.save_path = 'checkpoint'
    args.slide_win = 128#256
    args.slide_stride = 2
    args.hidden_size = 8
    args.batch = 16
    args.pred_len = 32#64
    args.seq_mask_range_low = 4
    args.seq_mask_range_high = 4
    # args.point_part = 8
    args.model_type = 'BiSeqMask'



    cfg = {
        'slide_win': args.slide_win,
        'slide_stride': args.slide_stride,
        'test_slide_win': int(args.slide_win+args.slide_win/4),  # add the active window
        'test_slide_stride': int(args.slide_win/4),
    }

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'seed': args.random_seed,
        'val_ratio': args.val_ratio,  # 0.1
        'decay': args.decay,
        'topk': args.topk,
        'lr': args.learning_rate
    }

    train_dataset = SNetDataset(data_dict['train'], mode='train', add_anomaly=False,
                                test_label=data_dict['test_labels'], config=cfg)
    train_dataloader, val_dataloader = get_loaders(train_dataset=train_dataset, seed=train_config['seed'],
                                                   batch=train_config['batch'],
                                                   val_ratio=train_config['val_ratio'])

    test_dataset = SNetDataset(data_dict['test'], mode='test',  test_slide = int(args.slide_win/4), test_label=data_dict['test_labels'], config=cfg)

    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False)  # feed each sample into


    #################################################################################################################
    part = [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]

    print('level number {}, level details: {}'.format(len(part), part))

    if args.model_type == 'BiSeqMask':
        print('Model_type: BiSeqMask')
        model = SCIMaskEvenPretrain(args, input_len=args.slide_win,
                                    seq_mask_range_ratio=[args.seq_mask_range_low, args.seq_mask_range_high],
                                    pred_len=[args.pred_len, args.pred_len],
                                    input_dim=args.variate_index,
                                    number_levels=len(part),
                                    number_level_part=part, num_layers=3)
    else:
        print('Model_type: BiPointMask')
        model = SCI_Point_Mask(args, num_classes=args.slide_win, input_len=args.slide_win, input_dim=3,
                               number_levels=len(part),
                               number_level_part=part, num_layers=3)

    #################################################################################################################
    dir_path = args.save_path

    # model_load_path = f'{dir_path}/NewDataBiEvenSeqMaskECG/{args.variate_index}dim/{args.dataset}_{args.model_type}_' \
    #                   f'Pretrain2Stack_{args.variate_index}dimNew_group1_lradj{args.lradj}_' \
    #                   f'{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_08_13_100824.pt',

    # model_load_path = f'{dir_path}/NewDataBiEvenSeqMaskECG/{args.variate_index}dim/' \
    #                   f'ECG_BiSeqMask_Pretrain2Stack_16dimNew_group1_lradj3_128_h4.0_bt32_p32_08_13_100824.pt'

    model_load_path = f'{dir_path}/NewDataBiEvenSeqMaskECG/{args.variate_index}dim/' \
                      f'ecg_BiSeqMask_Pretrain2Stack_2dimNew_group1_lradj3_128_h8_bt16_p32_08_17_230247.pt'   # chfdb_chf01_275_new

    # model_load_path = f'{dir_path}/mitdb__100_180/{args.variate_index}dim/' \
    #                       f'ecg_BiSeqMask_8_Pretrain_2dimNew_group1_lradj3_256_h8_bt32_p64_08_21_155419.pt'

    # model_load_path = f'{dir_path}/chfdb_chf13_45590/{args.variate_index}dim/' \
    #                       f'ecg_BiSeqMask_4_Pretrain_2dimNew_group1_lradj3_128_h8_bt32_p32_08_23_143923.pt'

    # model_load_path = f'{dir_path}/chfdbchf15/{args.variate_index}dim/' \
    #     #                   f'ecg_BiSeqMask_4_Pretrain_2dimNew_group1_lradj3_128_h8_bt32_p32_08_23_150113.pt'

    # model_load_path = f'{dir_path}/ltstdb_20221_43/{args.variate_index}dim/' \
    #                   f'ecg_BiSeqMask_4_Pretrain_2dimNew_group1_lradj3_128_h16_bt32_p32_08_23_155840.pt'

    # model_load_path = f'{dir_path}/ltstdb_20321_240/{args.variate_index}dim/' \
    #                   f'ecg_BiSeqMask_4_Pretrain_2dimNew_group1_lradj3_128_h16_bt32_p32_08_23_161530.pt'


########################################################################################################################
    # dir_path = 'ECG_GRU_Results'
    # model_load_path = f'{dir_path}/chfdb_chf01_275_new/{args.variate_index}dim/' \
    #                   f'ecg_BiSeqMask_4_Pretrain_2dimNew_group1_lradj3_128_h16_bt16_p32_08_29_110619.pt'

    # dir_path = 'ECG_TCN_Results'
    # model_load_path = f'{dir_path}/chfdb_chf01_275_new/{args.variate_index}dim/' \
    #                   f'ecg_BiSeqMask_4_Pretrain_2dimNew_group1_lradj3_128_h16_bt16_p32_08_28_223850.pt'

###New########################################################################################################################
    args.hidden_size = 16
    args.batch = 16
    args.groups = 1    #3.92
    model = SCIMaskEvenPretrain(args, input_len=args.slide_win,
                                seq_mask_range_ratio=[args.seq_mask_range_low, args.seq_mask_range_high],
                                pred_len=[args.pred_len, args.pred_len],
                                input_dim=args.variate_index,
                                number_levels=len(part),
                                number_level_part=part, num_layers=3)
    model_load_path = 'I:\Papers\AAAI2022anomaly\TAD_SCINet\TADib\ECG_Results\chfdb_chf01_275_new' \
                      '/2dim/ecg_BiSeqMask_4_Pretrain_2dimNew_group1_lradj3_128_h16_bt16_p32_09_12_210706.pt'  # chfdb_chf01_275_new
    # F1 score: 0.6342595978062158
    # precision: 0.6086956521739131
    # recall: 0.6222222222222222
    # F1 score: 0.7607515269736206, Threshods: 3.7
    # localization F1 score: 0.8089887640449438
    # AUROC: 0.8760493852226556, AUPRC: 0.8517566928773677


################################################################################
    # args.hidden_size = 16
    # args.batch = 16
    # args.groups = 2  # 3.92
    # model = SCIMaskEvenPretrain(args, input_len=args.slide_win,
    #                             seq_mask_range_ratio=[args.seq_mask_range_low, args.seq_mask_range_high],
    #                             pred_len=[args.pred_len, args.pred_len],
    #                             input_dim=args.variate_index,
    #                             number_levels=len(part),
    #                             number_level_part=part, num_layers=3)
    # model_load_path = 'I:\Papers\AAAI2022anomaly\TAD_SCINet\TADib\ECG_Results\chfdb_chf13_45590/' \
    #                   '2dim/ecg_BiSeqMask_4_Pretrain_2dimNew_group2_lradj3_128_h16_bt16_p32_09_13_135318.pt'  # chfdb_chf01_275_new

    # ecg_BiSeqMask_nhid8_bt16_lrtype3_pred32_group2.log
    # F1 score: 0.6230366492146596
    # precision: 0.536036036036036
    # recall: 0.7484276729559748
    # F1 score: 0.6984962399138224, Threshods: 10.7
    # F1 score: 0.7806267806267807
    # AUROC: 0.8440133878492998, AUPRC: 0.7547804752624407


################################################################################

    #
    # args.hidden_size = 16
    # args.batch = 16
    # args.groups = 1  # 3.92
    # model = SCIMaskEvenPretrain(args, input_len=args.slide_win,
    #                             seq_mask_range_ratio=[args.seq_mask_range_low, args.seq_mask_range_high],
    #                             pred_len=[args.pred_len, args.pred_len],
    #                             input_dim=args.variate_index,
    #                             number_levels=len(part),
    #                             number_level_part=part, num_layers=3)
    # model_load_path = 'I:\Papers\AAAI2022anomaly\TAD_SCINet\TADib\ECG_Results\chfdbchf15/' \
    #                   '2dim/ecg_BiSeqMask_4_Pretrain_2dimNew_group1_lradj3_128_h16_bt16_p32_09_13_163934.pt'  # chfdb_chf01_275_new

    # ecg_BiSeqMask_nhid8_bt16_lrtype3_pred32_group1
    # F1 score: 0.6254692556634305
    # precision: 0.5786163522012578
    # recall: 0.6174496644295302

    # F1 score: 0.7226142155373856, Threshods: 11.14
    # F1 score: 0.7836734693877551
    # AUROC: 0.9404689563498224, AUPRC: 0.7347106689056331


    #
    # args.hidden_size = 16
    # args.batch = 16
    # args.groups = 1  # 3.92
    # model = SCIMaskEvenPretrain(args, input_len=args.slide_win,
    #                             seq_mask_range_ratio=[args.seq_mask_range_low, args.seq_mask_range_high],
    #                             pred_len=[args.pred_len, args.pred_len],
    #                             input_dim=args.variate_index,
    #                             number_levels=len(part),
    #                             number_level_part=part, num_layers=3)
    # model_load_path = 'I:\Papers\AAAI2022anomaly\TAD_SCINet\TADib\ECG_Results\ltstdb_20221_43/' \
    #                   '2dim/ecg_BiSeqMask_4_Pretrain_2dimNew_group1_lradj3_128_h16.0_bt16_p32_09_14_004648.pt'  # chfdb_chf01_275_new
    #
    # # ecg_BiSeqMask_nhid16_bt16_lrtype3_pred32_group1
    # F1 score: 0.3774509803921568
    # precision: 0.2733812949640288
    # recall: 0.5891472868217055

    # AUROC: 0.7255250320092124
    # AUPRC: 0.23144886253025443

    # F1 score: 0.4021795606872942, Threshods: 1.683804817689596
    # F1 score: 0.4471403812824956







    print(model_load_path)
    model.load_state_dict(torch.load(model_load_path))
    best_model = model.to(args.device)

    loss_func = nn.L1Loss()
    device = args.device

    test_loss_list = []
    now = time.time()

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(test_dataloader)

    model.eval()

    # _, test_result = test(model, test_dataloader,
    #                       type=0)  # avg_loss, [test_predicted_list, test_ground_list, test_labels_list] 2034*27
    # get_score(test_result, test_result)


#################################################################################################
    Stride = int(args.slide_win / 4)
    ActiveWindow = args.slide_win
    find_threshold = False
    variate_num = [0, 1]
    number = data_dict['test_labels'].shape[0]
    count = -1
    threshod =  1.6838# 10 # 0.3
    threshod_l1 = 0.1
    NewLabels = np.zeros(number)
    TotalLabels = data_dict['test_labels']
    anomaly = np.where(TotalLabels == 1)[0]
    enter = False
    phi = 1
    ERRORS = []
    for x, y, labels in test_dataloader:
        x, y, labels = [item.to(device).float() for item in [x, y, labels]]
        count = count + 1

        with torch.no_grad():
            x_temp = x.detach().cpu().numpy()
            x_value = x[:, -ActiveWindow:, :]
            y_value = y[:, -ActiveWindow:, :]
            labels_value = labels[:, -ActiveWindow:]
            predicted = model(x_value)  # torch.Size([31, 27])

            l1Error = loss_func(predicted, y_value).detach().cpu().numpy()

            predicted = predicted[:, :, variate_num].squeeze().detach().cpu().numpy()
            if len(variate_num) > 1:
                smoothed_pred = []
                smoothed_gt = []
                for i in variate_num:
                    smoothed_pred.append(smooth(predicted[:, i], window_len=10)[0:-9])
                    smoothed_gt.append(smooth(y_value[:,:, i].squeeze().detach().cpu().numpy(), window_len=15)[0:-14])
                smoothed_pred = np.array(smoothed_pred).transpose()
                smoothed_gt = np.array(smoothed_gt).transpose()
                gt_seq = y_value[:, :, variate_num].squeeze().detach().cpu().numpy()
                dist_ts = dtw(smoothed_pred[-3 * Stride:, :], smoothed_gt[-3 * Stride:, :])
            else:
                smoothed_pred = smooth(predicted, window_len=5)[0:-4]
                gt_seq = y_value[:, :, variate_num].squeeze().detach().cpu().numpy()
                dist_ts = dtw(smoothed_pred[-3 * Stride:], gt_seq[-3 * Stride:])

            error_type = dist_ts
            ERRORS.append(error_type)
            # plt.plot(smoothed_pred, color='r')
            # plt.plot(gt_seq, color='b')
            # plt.title(f'SegDTWError:{dist_ts},L1Error: {l1Error}')
            # plt.plot(labels_value.squeeze().detach().cpu().numpy(), '.')
            # plt.show()
            if not find_threshold:
                End = Stride + Stride * count + ActiveWindow  # at the #160 point, the actual value is 159

                if error_type > phi * threshod:
                    print("Error occured")  # 32 +  0-127    32-159   64-191
                    enter = True

                    print('Start-->count{0}->{1}:{2}, error->{3}'.format(count, End - Stride, End, error_type))

                    NewLabels[End - Stride:   End] = 1  # find the last 32 dot should be the anomaly condidate
                    enumDtwError = []
                    enumL1Error = []
                    for k in range(Stride):
                        x_i = x[:, (-ActiveWindow - k):(ActiveWindow + Stride - k), :]  # -128 : 160 = 128        -127 : 159
                        y_i = y[:, (-ActiveWindow - k):(ActiveWindow + Stride - k), :]
                        labels_i = labels[:,
                                   (-ActiveWindow - k):(ActiveWindow + Stride - k)].squeeze().detach().cpu().numpy()
                        predicted_i = model(x_i)
                        l1Error_i = loss_func(predicted_i, y_i).detach().cpu().numpy()

                        predicted_i = predicted_i[:, :, variate_num].squeeze().detach().cpu().numpy()
                        if len(variate_num) > 1:
                            smoothed_pred_i = []
                            smoothed_gt_i = []
                            for i in variate_num:
                                smoothed_pred_i.append(smooth(predicted_i[:, i], window_len=3)[0:-2])
                                smoothed_gt_i.append(smooth(y_i[:,:, i].squeeze().detach().cpu().numpy(), window_len=3)[0:-2])
                            smoothed_pred_i = np.array(smoothed_pred_i).transpose()
                            smoothed_gt_i = np.array(smoothed_gt_i).transpose()
                        else:

                            smoothed_pred_i = smooth(predicted_i, window_len=5)[0:-4]
                        gt_seq_i = y_i[:, :, variate_num].squeeze().detach().cpu().numpy()
                        dist_ts_i = dtw(smoothed_pred_i[-3 * Stride:], smoothed_gt_i[-3 * Stride:])  ####################
                        enumDtwError.append(dist_ts_i)  #########################################  dtw
                        enumL1Error.append(l1Error_i)  #########################################  L1

                        # plt.plot(smoothed_pred_i[:,0], color='r', label = 'Reconstruction')
                        # plt.plot(gt_seq_i[:,0], color='b', label = 'Input')
                        # plt.plot(smoothed_pred_i[:, 1], color='r')
                        # plt.plot(gt_seq_i[:, 1], color='b')
                        # plt.title(f'SegDTWError:{dist_ts_i},L1Error: {l1Error_i}')
                        # plt.plot(labels_i*2, '.', label = 'Label')
                        # plt.legend(loc='lower left')
                        # plt.show()
                        # a = 0


                    pos = []
                    for j, error in enumerate(enumDtwError):
                        if error < 0.10 * phi * threshod:
                            pos.append(j)

                    if len(pos) == 0:
                        #     pos = np.where(enumDtwError == np.min(enumDtwError))
                        # else:
                        #     pos = pos[0]
                        print("All Error occured")
                    else:
                        pos = pos[0]                        # if
                        rpos = End - pos - (End-Stride)
                        for p in range(rpos):
                            print(" Error -> number remove:", End - Stride  + p)
                            # NewLabels[End - Stride:  End - Stride + p] = 0
                        NewLabels[End - Stride:  End - pos] = 0
                        a = 0

                #
                # else:
                #     if enter == True:  # outlier
                #         print("Error Exit,{0}".format(error_type))  # [///////////[//]...][AAAAA][AAAAA][AAAAA][AAAAA]
                #         enumDtwExitError = []
                #         enumlL1ExitError = []
                #         for k in range(Stride):
                #             x_i = x[:, (-ActiveWindow - k):(ActiveWindow + Stride - k), :]
                #             y_i = y[:, (-ActiveWindow - k):(ActiveWindow + Stride - k), :]
                #             labels_i = labels[:,
                #                        (-ActiveWindow - k):(ActiveWindow + Stride - k)].squeeze().detach().cpu().numpy()
                #             predicted_i = model(x_i)
                #             l1Error_i = loss_func(predicted_i, y_i).detach().cpu().numpy()
                #
                #             predicted_i = predicted_i[:, :, variate_num].squeeze().detach().cpu().numpy()
                #             if len(variate_num) > 1:
                #                 smoothed_pred_i = []
                #                 for i in variate_num:
                #                     smoothed_pred_i.append(smooth(predicted[:, i], window_len=5)[0:-4])
                #                 smoothed_pred_i = np.array(smoothed_pred_i).transpose()
                #             else:
                #
                #                 smoothed_pred_i = smooth(predicted_i, window_len=5)[0:-4]
                #
                #             gt_seq_i = y_i[:, :, variate_num].squeeze().detach().cpu().numpy()
                #             dist_ts_i = dtw(smoothed_pred_i[-3 * Stride:], gt_seq_i[-3 * Stride:])  ############
                #             enumDtwExitError.append(dist_ts_i)
                #             enumlL1ExitError.append(l1Error_i)
                #
                #             # plt.plot(smoothed_pred_i, color='r')
                #             # plt.plot(gt_seq_i, color='b')
                #             # plt.title(f'SegDTWError:{dist_ts_i},L1Error: {l1Error_i}')
                #             # plt.plot(labels_i, '.')
                #             # plt.show()
                #
                #         pos_exit = []
                #         for j, error in enumerate(enumDtwExitError):  # enumDtwError###################
                #             if error > 1.5*phi * threshod:
                #                 pos_exit.append(j)
                #
                #         if len(pos_exit) == 0:
                #             print("All Error Exit")
                #         else:
                #             pos = pos_exit[0]
                #             for p in range(pos):
                #                 NewLabels[End - ActiveWindow - p:End] = 0
                #
                #         enter = False


                # threshod = dist_ts
    if find_threshold:
        get_dtw(ERRORS, data_dict['test_labels'], 44900, interval=32)

    plt.plot(ERRORS, color='r')
    folder_path = args.save_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print(folder_path)
    np.savetxt(
        f'{folder_path}/NewDataBiEvenSeqMaskECG/{args.variate_index}dim/test_error.csv',
        ERRORS, delimiter=",")

    # plt.show()
    # NewLabels = np.zeros(number)
    # NewLabels[352:544] = 1
    # f1, precision, recall, TP, TN, FP, FN = calc_point2point(NewLabels, TotalLabels)
    f1 = f1_score(NewLabels, TotalLabels)
    print(f'Initial F1 score: {f1}')
    # print(f'Initial precision: {precision}')
    # print(f'Initial recall: {recall}\n')

    predict = adjust_predicts(score=TotalLabels, label=TotalLabels,
                              threshold=0,
                              pred=NewLabels,
                              calc_latency=False)

    # f1, precision, recall, TP, TN, FP, FN = calc_point2point(predict, TotalLabels)
    f1 = f1_score(NewLabels, TotalLabels)
    print(f'Adjusted F1 score: {f1}')
    # print(f'Adjusted precision: {precision}')
    # print(f'Adjusted recall: {recall}\n')
    # plt.plot(predict * 1.3, '.')
    # plt.plot(NewLabels * 1.2, '.')
    # plt.plot(TotalLabels, '.')
    # plt.show()

    # _, val_result = test(model, val_dataloader, type=0)
    # _, test_result = infer(model, test_dataloader,
    #                       type=0)  # avg_loss, [test_predicted_list, test_ground_list, test_labels_list] 2034*27
    #
    # # [test_predicted_list, test_ground_list, test_labels_list]
    # test_predicted_list = np.array(test_result[0])
    # test_ground_list = np.array(test_result[1])
    # test_labels_list = np.array(test_result[2])
    #
    # dtw_error = []
    #
    # for i in range(int(test_predicted_list.shape[0]/128)-1):
    #
    #     smoothed_pred = smooth(test_predicted_list[i*128:(i+1)*128, 0], window_len=5)[0:-4]
    #
    #     dist_ts = dtw(smoothed_pred, test_ground_list[i*128:(i+1)*128, 0])
    #     print(dist_ts)
    #     dtw_error.append(dist_ts)
    #
    #     plt.plot(smoothed_pred, color='r')
    #     plt.plot(test_ground_list[i*128:(i+1)*128, 0], color='b')
    #     plt.title(f'SegDTWError:{dist_ts}')
    #     plt.plot(test_labels_list[i*128:(i+1)*128], '.')
    #     plt.show()
    # plt.plot(dtw_error, color='r')
    # plt.show()
    # a = 0


