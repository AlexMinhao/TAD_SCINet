
import pandas as pd
import numpy as np
import torch
from torch import nn

from datasets.dataloader import load_dataset
import os
from datasets.SNetDataset import SNetDataset

import time
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import auc

from models.SCINetPWPretrainMask import SCI_Point_Mask
from models.SCINetBiEvenSeqPretrain import SCIMaskEvenPretrain
from models.TCN import TCN


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



def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 1:
        lr_adjust = {epoch: args.lr * (0.95 ** ((epoch - 1) // 1))}
    elif args.lradj == 2:

        lr_adjust = {
            20: 0.0005, 40: 0.0001, 60: 0.00005, 80: 0.00001

        }
    elif args.lradj == 3:

        lr_adjust = {
            20: 0.0005, 25: 0.0001, 35: 0.00005, 55: 0.00001
            , 70: 0.000001
        }
    elif args.lradj == 4:

        lr_adjust = {
            30: 0.0005, 40: 0.0003, 50: 0.0001, 65: 0.00001
            , 80: 0.000001
        }
    elif args.lradj == 5:

        lr_adjust = {
            40: 0.0001, 60: 0.00005
        }
    elif args.lradj == 6:

        lr_adjust = {
            0: 0.0001, 5: 0.0005, 10: 0.001, 20: 0.0001, 30: 0.00005, 40: 0.00001
            , 70: 0.000001
        }
    elif args.lradj == 61:

        lr_adjust = {
            0: 0.0001, 5: 0.0005, 10: 0.001, 25: 0.0005, 35: 0.0001, 50: 0.00001
            , 70: 0.000001
        }

    elif args.lradj == 7:

        lr_adjust = {
            10: 0.0001, 30: 0.00005, 50: 0.00001
            , 70: 0.000001
        }

    elif args.lradj == 8:

        lr_adjust = {
            0: 0.0005, 5: 0.0008, 10: 0.001, 20: 0.0001, 30: 0.00005, 40: 0.00001
            , 70: 0.000001
        }
    elif args.lradj == 9:

        lr_adjust = {
            0: 0.0001, 10: 0.0005, 20: 0.001, 40: 0.0001, 45: 0.00005, 50: 0.00001
            , 70: 0.000001
        }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))




def loss_func(y_pred, y_true):
    # loss = F.mse_loss(y_pred, y_true, reduction='mean')
    loss = nn.L1Loss()
    loss_value = loss(y_pred, y_true)
    return loss_value


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
    min_loss = 1e+8
    i = 0
    epoch = config['epoch']
    early_stop_win = 50

    model.train()
    stop_improve_count = 0

    dataloader = train_dataloader
    time_now = time.time()
    iter_count = 0
    for i_epoch in range(epoch):

        adjust_learning_rate(optimizer, i_epoch, args)
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
                if i_epoch > 10:
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
    if type == 1:
        folder_path = args.save_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print(folder_path)
        np.savetxt( #args.model_type
            f'{folder_path}/{subdataset}/{args.variate_index}dim/val_{args.dataset}_{args.model_type}_{args.point_part}_{args.variate_index}dimNum{args.variate_index}_group{args.groups}_lradj{args.lradj}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_ground_list.csv',
            t_test_ground_list.detach().cpu().numpy(), delimiter=",")
        np.savetxt(
            f'{folder_path}/{subdataset}/{args.variate_index}dim/val_{args.dataset}_{args.model_type}_{args.point_part}_{args.variate_index}dimNum{args.variate_index}_group{args.groups}_lradj{args.lradj}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_predicted_list.csv',
            t_test_predicted_list.detach().cpu().numpy(), delimiter=",")
        np.savetxt(
            f'{folder_path}/{subdataset}/{args.variate_index}dim/val_{args.dataset}_{args.model_type}_{args.point_part}_{args.variate_index}dimNum{args.variate_index}_group{args.groups}_lradj{args.lradj}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_labels_list.csv',
            t_test_labels_list.detach().cpu().numpy(), delimiter=",")


    elif type == 2:
        folder_path = args.save_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print(folder_path)
        np.savetxt(
            f'{folder_path}/{subdataset}/{args.variate_index}dim/test_{args.dataset}_{args.model_type}_{args.point_part}_{args.variate_index}dimNum{args.variate_index}_group{args.groups}_lradj{args.lradj}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_ground_list.csv',
            t_test_ground_list.detach().cpu().numpy(), delimiter=",")
        np.savetxt(
            f'{folder_path}/{subdataset}/{args.variate_index}dim/test_{args.dataset}_{args.model_type}_{args.point_part}_{args.variate_index}dimNum{args.variate_index}_group{args.groups}_lradj{args.lradj}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_predicted_list.csv',
            t_test_predicted_list.detach().cpu().numpy(), delimiter=",")
        np.savetxt(
            f'{folder_path}/{subdataset}/{args.variate_index}dim/test_{args.dataset}_{args.model_type}_{args.point_part}_{args.variate_index}dimNum{args.variate_index}_group{args.groups}_lradj{args.lradj}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_labels_list.csv',
            t_test_labels_list.detach().cpu().numpy(), delimiter=",")




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
    # if self.env_config['report'] == 'best':
    #     info = top1_best_info
    # elif self.env_config['report'] == 'val':
    #     info = top1_val_info

    # folder_path = './results/'

    # np.savetxt(f'{folder_path}/{args.dataset}/bestF1_BiSeqMask_pred_label.csv',
    #            top1_best_info[-1], delimiter=",")

    folder_path = args.save_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print(folder_path)
    np.savetxt(
        f'{folder_path}/{subdataset}/{args.variate_index}dim/bestF1_{args.dataset}_{args.model_type}_{args.point_part}_{args.variate_index}dimNum{args.variate_index}_group{args.groups}_lradj{args.lradj}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_pred_label.csv',
        top1_best_info[-1], delimiter=",")



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


    paths = [ #args.model_type
        f'{dir_path}/{subdataset}/{args.variate_index}dim/{args.dataset}_{args.model_type}_{args.point_part}_Pretrain_{args.variate_index}dimNew_group{args.groups}_lradj{args.lradj}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_{datestr}.pt',
        f'results/{dir_path}/{args.dataset}_{datestr}.csv',
    ]

    for path in paths:
        dirname = os.path.dirname(path)
        Path(dirname).mkdir(parents=True, exist_ok=True)

    return paths


if __name__ == "__main__":
    torch.manual_seed(4213)  # reproducible 4321  1234  2314 1324  4213
    torch.cuda.manual_seed_all(4213)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True
    # 'machine-1-1'
    args.dataset = 'ecg'    #chfdb_chf13_45590   chfdbchf15   ltstdb_20221_43  ltstdb_20321_240   mitdb__100_180
    subdataset = 'mitdb__100_180'
    if args.dataset == 'ecg':
        args.save_path = 'ECG_Results'
        args.variate_index = 2
        args.slide_win = 256#128#256
        args.slide_stride = 2
        # args.hidden_size = 16
        # args.batch = 16
        args.pred_len = 64#32#64
        args.seq_mask_range_low = 4
        args.seq_mask_range_high = 4
        args.point_part = 4
        args.model_type = 'BiSeqMask' #BiPointMask'#'BiSeqMask'

        args.epoch = 50

    data_dict = load_dataset(args.dataset, subdataset= subdataset, use_dim="all", root_dir="/", nrows=None)
    #add outliers
    # outliers = [835, 944, 1028, 1105 ,1162, 1242, 1285 ,1360 ,1442 ,1551,
    #             895, 994, 1098, 1195 ,1192, 1292, 1385 ,1320 ,1492 ,1591,]
    # for i in outliers:
    #     data_dict['test'][i, 0] = data_dict['test'][i, 0] * random.randint(1, 9)
    #     data_dict['test'][i, 1] = data_dict['test'][i, 1] * random.randint(1, 9)
    # data_dict['test'][865,0] = data_dict['test'][865,0]*random.randint(1,9)
    # data_dict['test'][944,1] = data_dict['test'][944,1]*random.randint(1,9)
    # data_dict['test'][1028,0] = data_dict['test'][1028,0]*random.randint(1,9)
    # data_dict['test'][1105,1] = data_dict['test'][1105,1]*random.randint(1,9)
    # data_dict['test'][1162,0] = data_dict['test'][1162,0]*random.randint(1,9)
    # data_dict['test'][1242,1] = data_dict['test'][1242,1]*random.randint(1,9)
    # data_dict['test'][1285,0] = data_dict['test'][1285,0]-random.randint(1,9)
    # data_dict['test'][1360,1] = data_dict['test'][1360,1]-random.randint(1,9)
    # data_dict['test'][1442,0] = data_dict['test'][1442,0]*random.randint(1,9)
    # data_dict['test'][1551,1] = data_dict['test'][1551,1]+random.randint(1,9)

    # data_dict['test_labels'][outliers] = 1
    #
    # data_dict['test'] = data_dict['test'][630:,:]
    # data_dict['test_labels']= data_dict['test_labels'][630:]

    # np.savetxt(
    #     f'./ECG_Results/swat_new_train.csv',
    # #     data_dict['train'], delimiter=",")
    # np.savetxt(
    #     f'./ECG_Results/ecg_outlier_test.csv',
    #     data_dict['test'], delimiter=",")
    # np.savetxt(
    #     f'./ECG_Results/ecg_outlier_label.csv',
    #     data_dict['test_labels'], delimiter=",")


    cfg = {
        'slide_win': args.slide_win,
        'slide_stride': args.slide_stride,
        'test_slide_win': args.slide_win,
        'test_slide_stride': args.slide_win,
    }

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'seed': args.random_seed,
        'val_ratio': args.val_ratio,  #0.1
        'decay': args.decay,
        'topk': args.topk,
        'lr': args.learning_rate
    }
 
    train_dataset = SNetDataset(data_dict['train'], mode='train', add_anomaly = False, test_label = data_dict['test_labels'], config=cfg)
    train_dataloader, val_dataloader = get_loaders(train_dataset = train_dataset, seed = train_config['seed'], batch = train_config['batch'],
                                                        val_ratio=train_config['val_ratio'])

    test_dataset = SNetDataset(data_dict['test'], mode='test', test_label = data_dict['test_labels'], config=cfg)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    part = [[1, 1], [1, 1],[1, 1], [0, 0], [0, 0],[0, 0], [0, 0]]

    print('level number {}, level details: {}'.format(len(part), part))

    if args.model_type == 'BiSeqMask':
        print('Model_type: BiSeqMask')
        model = SCIMaskEvenPretrain(args, input_len=args.slide_win, seq_mask_range_ratio=[args.seq_mask_range_low, args.seq_mask_range_high], pred_len=[args.pred_len, args.pred_len],
                                input_dim=args.variate_index,
                        number_levels=len(part),
                        number_level_part=part, num_layers=3)


    else:
        print('Model_type: BiPointMask')
        model = SCI_Point_Mask(args, num_classes=args.slide_win, input_len=args.slide_win, input_dim=args.variate_index,
                               number_levels=len(part),
                               number_level_part=part, point_part= args.point_part, num_layers=3)
    print(args)
    model.to(args.device)
    model_save_path = get_save_path()[0]
    args.train = 1
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


        _,val_result = test(model, val_dataloader, type=1)
        _,test_result = test(model, test_dataloader, type=2)  # avg_loss, [test_predicted_list, test_ground_list, test_labels_list] 2034*27
        get_score(test_result, val_result)


    else:
        dir_path = args.save_path

        # model_load_path = f'{dir_path}/{args.dataset}_best_' + \
        #                   f'SCINetBiSeqRollingScan_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_mrl{args.seq_mask_range_high}' \
        #                   f'_07_18_215607.pt'

        model_load_path = f'I:\\Papers\\AAAI2022anomaly\\TAD_SCINet\\TADib\\ECG_Results\\chfdb_chf01_275_new\\2dim\\' \
        f'ecg_BiPointMask_8_Pretrain_2dimNew_group1_lradj3_128_h8_bt32_p32_08_20_215143.pt',

        model.load_state_dict(torch.load(model_load_path))
        best_model = model.to(args.device)



        _, val_result = test(model, val_dataloader, type=0)
        _, test_result = test(model, test_dataloader,
                              type=0)  # avg_loss, [test_predicted_list, test_ground_list, test_labels_list] 2034*27
        get_score(test_result, val_result)




