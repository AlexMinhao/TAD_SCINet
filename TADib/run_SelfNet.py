from scipy.stats import rankdata, iqr, trim_mean
import pandas as pd
import numpy as np
import torch
from torch import nn
from datasets.data_preprocess import preprocessor,generate_windows_with_index
from datasets.dataloader import load_dataset
import os
from datasets.SNetDataset import SNetDataset
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F
from models.SCINet import SCINet
# from models.SCIMask import SCIMask
# from models.SCINetOneBranchMask import SCIMask
from models.SCINetEnsemble import SCIMask
from models.SCINetBiSeqMask import SCIMask
# from models.SCINetBiSeqMaskReScan import SCIMask
# from models.SCINetNewEnsemble import SCIMask
# from models.SCINetBiSeqRollingScan import SCIMask
from models.TCN import TCN
from models.GRU import GRU

from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores
from torch.utils.data import DataLoader, random_split, Subset
import random
import argparse
from datetime import datetime
from pathlib import Path
import math

parser = argparse.ArgumentParser()

parser.add_argument('--batch', help='batch size', type = int, default=16)
parser.add_argument('--epoch', help='train epoch', type = int, default=100)
parser.add_argument('--learning_rate', help='lr', type = float, default=0.001)

parser.add_argument('--slide_win', help='slide_win', type = int, default=768)

parser.add_argument('--slide_stride', help='slide_stride', type = int, default=32)
parser.add_argument('--save_path_pattern', help='save path pattern', type = str, default='msl')
parser.add_argument('--dataset', help='wadi / swat', type = str, default='')
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

parser.add_argument('--hidden-size', default=3, type=int, help='hidden channel of module')
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
parser.add_argument('--kernel', default=3, type=int, help='kernel size')
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--save_path', type=str, default='checkpoint')

parser.add_argument('--pred_len', type=int, default=128)
parser.add_argument('--seq_mask_range_low', type=int, default=8)
parser.add_argument('--seq_mask_range_high', type=int, default=4)

parser.add_argument('--ensemble', type=int, default=0)

parser.add_argument('--model_name', type=str, default='SCINet')

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')




def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSincePlus(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(model=None, save_path=None, config={}, train_dataloader=None, val_dataloader=None,test_dataloader=None):
    seed = config['seed']

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['decay'])

    now = time.time()

    train_loss_list = []
    cmp_loss_list = []

    device = args.device

    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    early_stop_win = 50

    model.train()


    stop_improve_count = 0

    dataloader = train_dataloader
    time_now = time.time()
    iter_count = 0
    for i_epoch in range(epoch):

        acu_loss = 0
        model.train()
        #   feature, y, label, edge_index
        for x, y, attack_labels in dataloader:
            _start = time.time()

            x, y = [item.float().to(device) for item in [x, y]]

            # x = x.permute(0,2,1) # torch.Size([128, 16, 27])
            #SMAP: ([16, 128, 25])

            optimizer.zero_grad()
            out = model(x)

            if isinstance(out,tuple):
                # rec = out[0].float().to(device)
                # mask = out[1]
                # masked_y = y.masked_fill(mask, 0)
                # loss = loss_func(rec, masked_y)

                loss = loss_func(out[0], y) #+ loss_func(out[1], y)

            else:
                loss = loss_func(out, y)

            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            acu_loss += loss.item()
            iter_count = iter_count+1
            i += 1
            if (i + 1) % 10 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, i_epoch + 1, loss.item()))
                speed = (time.time() - time_now) / i
                print('\tspeed: {:.4f}s/iter'.format(speed))
                iter_count = 0
                time_now = time.time()
        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
            i_epoch, epoch,
            acu_loss / len(dataloader), acu_loss), flush=True
        )

        # use val dataset to judge
        if val_dataloader is not None:
            if args.ensemble:
                val_loss, val_result_0, val_result_1 = testEnsemble(model, val_dataloader)  # Val
            else:
                val_loss, val_result = test(model, val_dataloader)  # Val

            print('epoch ({} / {}) (val_loss --- Loss:{:.8f}'.format(
                i_epoch, epoch,
                val_loss), flush=True
            )

            if val_loss < min_loss:
                if i_epoch > 30:
                    _, test_result = test(model, test_dataloader,
                                          type=2)  # avg_loss, [test_predicted_list, test_ground_list, test_labels_list] 2034*27
                    get_score(test_result, test_result)

                min_loss = val_loss
                stop_improve_count = 0
                torch.save(model.state_dict(), save_path)
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss:

                min_loss = acu_loss

    return train_loss_list

def testEnsemble(model, dataloader, type = 0):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = args.device

    test_loss_list = []
    now = time.time()

    test_predicted_0_list = []
    test_ground_0_list = []
    test_labels_0_list = []

    test_predicted_1_list = []
    test_ground_1_list = []
    test_labels_1_list = []

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

            predicted = model(x,y)  # torch.Size([31, 27])


            if isinstance(predicted,tuple):

                loss = loss_func(predicted, y) #+ loss_func(predicted[1], y)
            else:
                predicted = predicted.float().to(device)
                loss = loss_func(predicted, y)

            # labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])  # torch.Size([31, 27])

            if len(test_predicted_0_list) <= 0:
                test_predicted_0_list = predicted[0]
                test_ground_0_list = y
                test_labels_0_list = labels
            else:
                test_predicted_0_list = torch.cat((test_predicted_0_list, predicted[0]), dim=0)
                test_ground_0_list = torch.cat((test_ground_0_list, y), dim=0)
                test_labels_0_list = torch.cat((test_labels_0_list, labels), dim=0)

            if len(test_predicted_1_list) <= 0:
                test_predicted_1_list = predicted[1]
                test_ground_1_list = y
                test_labels_1_list = labels
            else:
                test_predicted_1_list = torch.cat((test_predicted_1_list, predicted[1]), dim=0)
                test_ground_1_list = torch.cat((test_ground_1_list, y), dim=0)
                test_labels_1_list = torch.cat((test_labels_1_list, labels), dim=0)

        test_loss_list.append(loss.item())
        acu_loss += loss.item()

        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    # t_test_predicted_list=t_test_predicted_list.permute(0,2,1)
    data_dim = test_predicted_0_list.shape[2]
    test_predicted_0_list = test_predicted_0_list.reshape(-1,data_dim)
    test_ground_0_list = test_ground_0_list.reshape(-1, data_dim)

    test_predicted_1_list = test_predicted_1_list.reshape(-1, data_dim)
    test_ground_1_list = test_ground_1_list.reshape(-1, data_dim)

    test_labels_0_list = test_labels_0_list.reshape(-1, 1)  #torch.Size([2048, 1])
    if type == 1:
        folder_path = './results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print(folder_path)


        np.savetxt(f'{folder_path}/{args.dataset}/val_6pNOShareEnsemble_RandSeq_0_ground_list.csv', test_ground_0_list.detach().cpu().numpy(), delimiter=",")
        np.savetxt(f'{folder_path}/{args.dataset}/val_6pNOShareEnsemble_RandSeq_0_predicted_list.csv', test_predicted_0_list.detach().cpu().numpy(), delimiter=",")
        np.savetxt(f'{folder_path}/{args.dataset}/val_6pNOShareEnsemble_RandSeq_0_labels_list.csv', test_labels_0_list.detach().cpu().numpy(), delimiter=",")

        np.savetxt(f'{folder_path}/{args.dataset}/val_6pNOShareEnsemble_RandSeq_1_ground_list.csv',
                   test_ground_1_list.detach().cpu().numpy(), delimiter=",")
        np.savetxt(f'{folder_path}/{args.dataset}/val_6pNOShareEnsemble_RandSeq_1_predicted_list.csv',
                   test_predicted_1_list.detach().cpu().numpy(), delimiter=",")
        np.savetxt(f'{folder_path}/{args.dataset}/val_6pNOShareEnsemble_RandSeq_1_labels_list.csv',
                   test_labels_0_list.detach().cpu().numpy(), delimiter=",")
    elif type == 2:
        folder_path = './results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print(folder_path)

        np.savetxt(f'{folder_path}/{args.dataset}/test_6pNOShareEnsemble_RandSeq_0_ground_list.csv', test_ground_0_list.detach().cpu().numpy(), delimiter=",")
        np.savetxt(f'{folder_path}/{args.dataset}/test_6pNOShareEnsemble_RandSeq_0_predicted_list.csv', test_predicted_0_list.detach().cpu().numpy(), delimiter=",")
        np.savetxt(f'{folder_path}/{args.dataset}/test_6pNOShareEnsemble_RandSeq_0_labels_list.csv', test_labels_0_list.detach().cpu().numpy(), delimiter=",")

        np.savetxt(f'{folder_path}/{args.dataset}/test_6pNOShareEnsemble_RandSeq_1_ground_list.csv', test_ground_1_list.detach().cpu().numpy(), delimiter=",")
        np.savetxt(f'{folder_path}/{args.dataset}/test_6pNOShareEnsemble_RandSeq_1_predicted_list.csv', test_predicted_1_list.detach().cpu().numpy(), delimiter=",")
        np.savetxt(f'{folder_path}/{args.dataset}/test_6pNOShareEnsemble_RandSeq_1_labels_list.csv', test_labels_0_list.detach().cpu().numpy(), delimiter=",")



    test_predicted_0_list = test_predicted_0_list.tolist()
    test_ground_0_list = test_ground_0_list.tolist()
    test_labels_0_list = test_labels_0_list.tolist()


    test_predicted_1_list = test_predicted_1_list.tolist()
    test_ground_1_list = test_ground_1_list.tolist()
    test_labels_1_list = test_labels_0_list

    avg_loss = sum(test_loss_list) / len(test_loss_list)




    return avg_loss, [test_predicted_0_list, test_ground_0_list, test_labels_0_list], [test_predicted_1_list, test_ground_1_list, test_labels_1_list]

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
            # x = x.permute(0, 2, 1)  # torch.Size([128, 16, 27])
            # y = y.permute(0, 2, 1)  # torch.Size([128, 16, 27])

            predicted = model(x)  # torch.Size([31, 27])


            if isinstance(predicted,tuple):
                # rec = predicted[0].float().to(device)
                # mask = predicted[1]
                # masked_y = y.masked_fill(mask, 0)
                # unmaske_y = (mask == False)
                # loss = loss_func(rec, masked_y)
                #
                # anti_masked_y = y.masked_fill(unmaske_y, 0)
                # predicted = rec + anti_masked_y
                mask = predicted[1]
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
                #
                # t_test_mask_list = mask
                # t_test_input_list = x
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

                # t_test_mask_list = torch.cat((t_test_mask_list,mask), dim=0)
                # t_test_input_list = torch.cat((t_test_input_list, x), dim=0)

        test_loss_list.append(loss.item())
        acu_loss += loss.item()

        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    # t_test_predicted_list=t_test_predicted_list.permute(0,2,1)
    data_dim = t_test_predicted_list.shape[2]

    # t_test_mask_list = t_test_mask_list.reshape(-1,data_dim)
    # t_test_input_list = t_test_input_list.reshape(-1,data_dim)

    t_test_predicted_list = t_test_predicted_list.reshape(-1,data_dim)

    # t_test_ground_list = t_test_ground_list.permute(0, 2, 1)
    t_test_ground_list = t_test_ground_list.reshape(-1, data_dim)


    t_test_labels_list = t_test_labels_list.reshape(-1, 1)  #torch.Size([2048, 1])
    if type == 1:
        folder_path = './results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print(folder_path)

        np.savetxt(f'{folder_path}/{args.dataset}/val_{args.model_name}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_mrl{args.seq_mask_range_high}_ground_list.csv', t_test_ground_list.detach().cpu().numpy(), delimiter=",")
        np.savetxt(f'{folder_path}/{args.dataset}/val_{args.model_name}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_mrl{args.seq_mask_range_high}_predicted_list.csv', t_test_predicted_list.detach().cpu().numpy(), delimiter=",")
        np.savetxt(f'{folder_path}/{args.dataset}/val_{args.model_name}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_mrl{args.seq_mask_range_high}_labels_list.csv', t_test_labels_list.detach().cpu().numpy(), delimiter=",")
        # np.savetxt(
        # f'{folder_path}/{args.dataset}/val_{args.model_name}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_mrl{args.seq_mask_range_high}_input_list.csv',
        # t_test_input_list.detach().cpu().numpy(), delimiter=",")
        # np.savetxt(
        # f'{folder_path}/{args.dataset}/val_{args.model_name}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_mrl{args.seq_mask_range_high}_mask_list.csv',
        # t_test_mask_list.detach().cpu().numpy(), delimiter=",")

    elif type == 2:
        folder_path = './results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print(folder_path)

        np.savetxt(f'{folder_path}/{args.dataset}/test_{args.model_name}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_mrl{args.seq_mask_range_high}_ground_list.csv', t_test_ground_list.detach().cpu().numpy(), delimiter=",")
        np.savetxt(f'{folder_path}/{args.dataset}/test_{args.model_name}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_mrl{args.seq_mask_range_high}_predicted_list.csv', t_test_predicted_list.detach().cpu().numpy(), delimiter=",")
        np.savetxt(f'{folder_path}/{args.dataset}/test_{args.model_name}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_mrl{args.seq_mask_range_high}_labels_list.csv', t_test_labels_list.detach().cpu().numpy(), delimiter=",")
        print(f'{folder_path}/{args.dataset}/test_{args.model_name}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_mrl{args.seq_mask_range_high}')
        # np.savetxt(
        # f'{folder_path}/{args.dataset}/test_{args.model_name}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_mrl{args.seq_mask_range_high}_input_list.csv',
        # t_test_input_list.detach().cpu().numpy(), delimiter=",")
        # np.savetxt(
        # f'{folder_path}/{args.dataset}/test_{args.model_name}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_mrl{args.seq_mask_range_high}_mask_list.csv',
        # t_test_mask_list.detach().cpu().numpy(), delimiter=",")


    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()
    test_labels_list = t_test_labels_list.tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)




    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]


def get_lookback_window_score(slide_win, test_result,val_result):
    np_test_result = np.array(test_result)  # 3 2034 27
    np_val_result = np.array(val_result)  # 3*31*27
    total_time_len = len(test_result[0])
    rang = range(slide_win, total_time_len, slide_win) #2832
    Total_error_0 = np.zeros((2832, 3000))
    Total_error_1 = np.zeros((2832, 3000))
    Total_labels = np.zeros((2832, 3000))
    index = 0
    for i in rang:
        pred = test_result[0][i - slide_win:i]  # 27  15
        gt = test_result[1][i - slide_win:i]  # 27  15
        label = test_result[2][i - slide_win:i]
        np_test_label = np.array(label).transpose()
        np_test_pred = np.array(pred)
        np_test_gt = np.array(gt)

        seg_test_result = [pred,gt,label]
        # np_test_pred = np.array(pred)
        # np_test_gt = np.array(gt)
        # np_test_label = np.array(label)



        test_delta = np.abs(np.subtract(
            np.array(np_test_pred).astype(np.float64),
            np.array(np_test_gt).astype(np.float64)
        ))

        # test_scores, normal_scores = get_full_err_scores(seg_test_result, seg_test_result)

        # test_labels = seg_test_result[2]
        # top1_best_info = get_best_performance_data(test_scores, test_labels,
        #                                            topk=1) #1 3000   list
        Total_error_0[index, index:index+slide_win] = test_delta[:,0]
        Total_error_1[index, index:index + slide_win] = test_delta[:, 1]
        Total_labels[index, index:index+slide_win] = np_test_label
        index = index + 1

    Labels = []
    Erros0 = []
    Erros1 = []
    Values = []
    Value_reshape= np.zeros((168,3000))
    for k in range(2998):
        lb = max(Total_labels[:,k])
        Labels.append(lb)
        error0 = Total_error_0[:,k]
        nonzero = np.nonzero(error0)
        error0 = error0[nonzero]
        error0 = np.min(error0)
        Erros0.append(error0)

        error1 = Total_error_1[:, k]
        nonzero = np.nonzero(error1)
        error1 = error1[nonzero]
        error1 = np.min(error1)
        Erros1.append(error1)

        Values.append(error0+error1)

    all_scores = None
    epsilon = 1e-2
    Erros0 = np.array(Erros0)
    err_median_0 = np.median(Erros0)
    err_iqr_0 = iqr(Erros0)
    err_scores_0 = (Erros0 - err_median_0) / (np.abs(err_iqr_0) + epsilon)
    smoothed_err_scores_0 = np.zeros(err_scores_0.shape)
    before_num = 3
    for i in range(before_num, len(err_scores_0)):  # 3 2034
        temp = err_scores_0[i - before_num:i + 1]
        smoothed_err_scores_0[i] = np.mean(err_scores_0[i - before_num:i + 1])

    if all_scores is None:
        all_scores = smoothed_err_scores_0  # smoothed_err_scores

    else:
        all_scores = np.vstack((
            all_scores,
            smoothed_err_scores_0  # smoothed_err_scores
        ))


    Erros1 = np.array(Erros1)
    err_median_1 = np.median(Erros1)
    err_iqr_1 = iqr(Erros1)
    err_scores_1 = (Erros1 - err_median_1) / (np.abs(err_iqr_1) + epsilon)
    smoothed_err_scores_1 = np.zeros(err_scores_1.shape)
    before_num = 3
    for i in range(before_num, len(err_scores_1)):  # 3 2034
        temp = err_scores_1[i - before_num:i + 1]
        smoothed_err_scores_1[i] = np.mean(err_scores_1[i - before_num:i + 1])

    all_scores = np.vstack((
        all_scores,
        smoothed_err_scores_1  # smoothed_err_scores
    ))
    all_scores = np.sum(all_scores, axis=0)
    all_scores = all_scores[np.newaxis, :]

    top1_best_info = get_best_performance_data(all_scores, Labels,  # 1 2856   2856
                                               topk=1)


    # Values = np.array(Values)
    # Values = Values[np.newaxis, :]
    #
    # top1_best_info = get_best_performance_data(Values, Labels,
    #                                            topk=1) #1 3000   list
    print(f'F1 score: {top1_best_info[0]}')
    print(f'precision: {top1_best_info[1]}')
    print(f'recall: {top1_best_info[2]}\n')


    # folder_path = './post_processing/'
    # np.savetxt(f'{folder_path}/{args.dataset}/{args.model_name}_slid_label_map.csv',
    #            Total_labels, delimiter=",")
    # np.savetxt(f'{folder_path}/{args.dataset}/{args.model_name}_slid_socre_map.csv',
    #            Total_scores, delimiter=",")
    a = 0



def get_score(test_result, val_result):

    feature_num = len(test_result[0][0])
    np_test_result = np.array(test_result)  # 3 2034 27
    np_val_result = np.array(val_result)  # 3*31*27

    test_labels = np_test_result[2, :].tolist()

    test_scores, normal_scores = get_full_err_scores(test_result, val_result)  # 27 2034  27 31

    top1_best_info = get_best_performance_data(test_scores, test_labels, #1 2856   2856
                                               topk=1)  # max(final_topk_fmeas), pre, rec, auc_score, thresold, pred_labels
    top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)

    print('=========================** Result **============================\n')

    info = None
    info = top1_best_info
    # if self.env_config['report'] == 'best':
    #     info = top1_best_info
    # elif self.env_config['report'] == 'val':
    #     info = top1_val_info

    folder_path = './results/'

    np.savetxt(f'{folder_path}/{args.dataset}/bestF1_BiSeqMask_pred_label.csv',
               top1_best_info[-1], delimiter=",")
    # dir_path =  './results/'
    # label_path = f'{dir_path}/{args.dataset}/' + \
    #                   f'{args.model_name}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_mrl{args.seq_mask_range_high}' \
    #                   f'pred_label.csv'
    # np.savetxt(label_path,top1_best_info[-1], delimiter=",")

    print(f'F1 score: {info[0]}')
    print(f'precision: {info[1]}')
    print(f'recall: {info[2]}\n')

def get_score_Ensemble(test_result, test_result_1):

    feature_num = len(test_result[0][0])
    np_test_result = np.array(test_result)  # 3 2034 27
    np_test_result_1 = np.array(test_result_1)


    test_labels = np_test_result[2, :].tolist()

    test_scores, normal_scores = get_full_err_scores(test_result, test_result)  # 27 2034  27 31

    test_scores_1, normal_scores_1 = get_full_err_scores(test_result_1, test_result_1)

    test_scores = test_scores+test_scores_1

    top1_best_info = get_best_performance_data(test_scores, test_labels,
                                               topk=1)  # max(final_topk_fmeas), pre, rec, auc_score, thresold, pred_labels


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


def get_save_path(feature_name=''):
    dir_path = args.save_path


    now = datetime.now()
    datestr = now.strftime('%m_%d_%H%M%S')


    paths = [
        f'{dir_path}/{args.model_name}/{args.dataset}_{args.model_name}_dim1_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_mrl{args.seq_mask_range_high}_{datestr}.pt',
        f'results/{dir_path}/{args.dataset}_{datestr}.csv',
    ]

    for path in paths:
        dirname = os.path.dirname(path)
        Path(dirname).mkdir(parents=True, exist_ok=True)

    return paths


if __name__ == "__main__":
    torch.manual_seed(4321)  # reproducible
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True
    # 'machine-1-1'
    args.dataset = 'gesture'
    args.slide_win = 168
    args.hidden_size = 8
    args.batch = 16
    args.pred_len = 48
    args.seq_mask_range_high = 1                #chfdb_chf13_45590
    args.model_name = 'BiSeqMask'

    data_dict = load_dataset(args.dataset, subdataset= 'mitdb__100_180', use_dim="all", root_dir="/", nrows=None)
    #train 496800 40        449919 40
    # pp = preprocessor()
    # data_dict = pp.normalize(data_dict)

    # generate sliding windows
    # window_dict = generate_windows_with_index(
    # data_dict, window_size=args.slide_win, stride=args.slide_stride
    # )
    #
    # train = window_dict["train_windows"]
    # test = window_dict["test_windows"]
    # test_labels = window_dict["test_labels"]
    # index = window_dict["index_windows"]

    # data_dict = load_dataset('msl', subdataset=None, use_dim="all", root_dir="/", nrows=None)

    cfg = {
        'slide_win': args.slide_win,
        'slide_stride': args.slide_stride,
    }

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'seed': args.random_seed,
        'val_ratio': args.val_ratio,
        'decay': args.decay,
        'topk': args.topk,
        'lr': args.learning_rate
    }
    # SWAT: TRAIN 496800   Test: 449919,   dim : 40   # 49680  44991

    # SMAP : TRAIN 135183,25   Test: 427617, 25
    # MSL : TRAIN 58317,55   Test: 73729, 55

    # gesture: TRAIN 8251   Test: 3000    dim: 2
    # power: TRAIN  18145   Test: 14786     zhouqi
    # nyc_taxi: TRAIN  13104   Test: 4416  dim: 3

    # kdd99: TRAIN 562387   Test 494021   dim: 34

    # data_dict['train'] = data_dict['train'][:, 25:40]
    # data_dict['test'] = data_dict['test'][:, 25:40]
    # data_dict['test_labels'] = data_dict['test_labels']
    train_dataset = SNetDataset(data_dict['train'], mode='train', add_anomaly = False, test_label = data_dict['test_labels'], config=cfg)
    train_dataloader, val_dataloader = get_loaders(train_dataset = train_dataset, seed = train_config['seed'], batch = train_config['batch'],
                                                        val_ratio=train_config['val_ratio'])


    test_dataset = SNetDataset(data_dict['test'], mode='test', test_label = data_dict['test_labels'], config=cfg)
    # folder_path = './results/'
    # np.savetxt(f'{folder_path}/{args.dataset}/test_dataset.csv', data_dict['test'],
    #            delimiter=",")
    # np.savetxt(f'{folder_path}/{args.dataset}/test_dataset_label.csv', data_dict['test_labels'],
    #            delimiter=",")

    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    print(args)

    part = [[1, 1], [1, 1],[1, 1], [0, 0], [0, 0],[0, 0], [0, 0]]

    print('level number {}, level details: {}'.format(len(part), part))
    # args.pred_len = 32

    # model = SCIMask(args, input_len=168, seg_slid=128, len_seg=41, seq_mask_range=[8, 6], pred_len=[48, 48],
    #                 MaskRange=[168 / 2, 168 / 12], input_dim=2,
    #                 number_levels=len(part),
    #                 pretrained=True, number_level_part=part, num_layers=3)

    model = SCIMask(args, input_len=168, seq_mask_range=[8, 6], pred_len=[args.pred_len, args.pred_len], input_dim=2,
                    number_levels=len(part),
                    number_level_part=part, num_layers=3)

    # channel_sizes = [8] * 3
    # model = TCN(input_size = data_dict['dim'], output_len=args.slide_win, num_channels=channel_sizes, kernel_size=args.kernel,
    #             dropout=args.dropout)

    # model = GRU(hidC = 2, hidR = 8)

    model.to(args.device)
    model_save_path = get_save_path()[0]
    args.train = 0
    args.ensemble = 0
    if args.train:
        train(model,
              save_path= model_save_path,
              config=train_config,
              train_dataloader=train_dataloader,
              val_dataloader=val_dataloader,
              test_dataloader =test_dataloader)

        model.load_state_dict(torch.load(model_save_path))
        best_model = model.to(args.device)
        #
        if args.ensemble:
            _, val_0_result, val_1_result = testEnsemble(model, val_dataloader, type=1)
            _, test_0_result, test_1_result = testEnsemble(model, test_dataloader,
                                                           type=2)  # avg_loss, [test_predicted_list, test_ground_list, test_labels_list] 2034*27

            get_score(test_0_result, val_0_result)
            get_score(test_1_result, val_1_result)
        else:
            _,val_result = test(model, val_dataloader, type=1)
            _,test_result = test(model, test_dataloader, type=2)  # avg_loss, [test_predicted_list, test_ground_list, test_labels_list] 2034*27
            get_score(test_result, val_result)


    else:
        dir_path = args.save_path

        model_load_path = f'{dir_path}/{args.model_name}/{args.dataset}_' + \
                          f'{args.model_name}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_mrl{args.seq_mask_range_high}' \
                          f'_07_22_093837.pt'


        model.load_state_dict(torch.load(model_load_path))
        best_model = model.to(args.device)


        if args.ensemble:
            _, val_0_result, val_1_result = testEnsemble(model, val_dataloader, type=1)
            _, test_0_result, test_1_result = testEnsemble(model, test_dataloader, type=2)


            #
            get_score(test_0_result, val_0_result)
            get_score(test_1_result, val_1_result)

            get_score_Ensemble(test_0_result,test_1_result)
        else:
            _, val_result = test(model, val_dataloader, type=1)
            _, test_result = test(model, test_dataloader,
                                  type=2)  # avg_loss, [test_predicted_list, test_ground_list, test_labels_list] 2034*27

            # get_lookback_window_score(168,test_result, val_result)
            get_score(test_result, val_result)




