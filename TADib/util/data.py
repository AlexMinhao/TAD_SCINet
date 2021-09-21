# util functions about data

from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import f1_score, mean_squared_error
import numpy as np
from numpy import percentile
from sklearn import metrics
from sklearn.metrics import auc


def get_attack_interval(attack): 
    heads = []
    tails = []
    for i in range(len(attack)):
        if attack[i] == 1:
            if attack[i-1] == 0:
                heads.append(i)
            
            if i < len(attack)-1 and attack[i+1] == 0:
                tails.append(i)
            elif i == len(attack)-1:
                tails.append(i)
    res = []
    for i in range(len(heads)):
        res.append((heads[i], tails[i]))
    # print(heads, tails)
    return res

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    predict = 1*predict
    actual = np.array(actual).squeeze()
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


# calculate F1 scores
def eval_scores(scores, true_scores, th_steps, return_thresold=False):

    padding_list = [0]*(len(true_scores) - len(scores))
    # print(padding_list)
    labels = np.array(true_scores)
    outlier = np.where(labels == 1.0)

    if len(padding_list) > 0:
        scores = padding_list + scores

    scores_sorted = rankdata(scores, method='ordinal') #返回的是这个数在序列中的排序位置
    th_steps = th_steps
    # th_steps = 500
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    fmeas = [None] * th_steps
    thresholds = [None] * th_steps
    TPs = []
    FPs = []
    Pres = []
    Recs = []

    for i in range(th_steps):
        temp = th_vals[i] * len(scores)
        cur_pred = scores_sorted > th_vals[i] * len(scores)

        fmeas[i] = f1_score(true_scores, cur_pred)

        temp1 = int(th_vals[i] * len(scores)+1)
        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores)+1))
        thresholds[i] = scores[score_index]

        f1, precision, recall, TP, TN, FP, FN = calc_point2point(cur_pred, true_scores)
        TPR = TP / (TP + FN + 0.00001)
        FPR = FP / (FP + TN + 0.00001)
        TPs.append(TPR)
        FPs.append(FPR)
        Pres.append(precision)
        Recs.append(recall)

    AUROC = metrics.auc(FPs, TPs)
    AUPRC = metrics.auc(Recs, Pres)

    print(f'AUROC : {AUROC}')
    print(f'AUPRC: {AUPRC}')


    if return_thresold:
        return fmeas, thresholds
    return fmeas

def eval_mseloss(predicted, ground_truth):

    ground_truth_list = np.array(ground_truth)
    predicted_list = np.array(predicted)

    
    # mask = (ground_truth_list == 0) | (predicted_list == 0)

    # ground_truth_list = ground_truth_list[~mask]
    # predicted_list = predicted_list[~mask]

    # neg_mask = predicted_list < 0
    # predicted_list[neg_mask] = 0

    # err = np.abs(predicted_list / ground_truth_list - 1)
    # acc = (1 - np.mean(err))

    # return loss
    loss = mean_squared_error(predicted_list, ground_truth_list)

    return loss

def get_err_median_and_iqr(predicted, groundtruth):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)

    # ** 四分位数 ** ** 概念 **：把给定的乱序数值由小到大排列并分成四等份，处于三个分割点位置的数值就是四分位数。 ** 第1四分位数(
    #     Q1) **，又称“较小四分位数”，等于该样本中所有数值由小到大排列后第25 % 的数字。 ** 第2四分位数(Q2) **，又称“中位数”，等于该样本中所有数值由小到大排列后第50 % 的数字。 ** 第3四分位数(
    #     Q3) **，又称“较大四分位数”，等于该样本中所有数值由小到大排列后第75 % 的数字。 ** 四分位距 **（InterQuartile
    # Range, IQR）= 第3四分位数与第1四分位数的差距

    return err_median, err_iqr

def get_err_median_and_quantile(predicted, groundtruth, percentage):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    # err_iqr = iqr(np_arr)
    err_delta = percentile(np_arr, int(percentage*100)) - percentile(np_arr, int((1-percentage)*100))

    return err_median, err_delta

def get_err_mean_and_quantile(predicted, groundtruth, percentage):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = trim_mean(np_arr, percentage)
    # err_iqr = iqr(np_arr)
    err_delta = percentile(np_arr, int(percentage*100)) - percentile(np_arr, int((1-percentage)*100))

    return err_median, err_delta

def get_err_mean_and_std(predicted, groundtruth):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_mean = np.mean(np_arr)
    err_std = np.std(np_arr)

    return err_mean, err_std


def get_f1_score(scores, gt, contamination):

    padding_list = [0]*(len(gt) - len(scores))
    # print(padding_list)

    threshold = percentile(scores, 100 * (1 - contamination))

    if len(padding_list) > 0:
        scores = padding_list + scores

    pred_labels = (scores > threshold).astype('int').ravel()

    return f1_score(gt, pred_labels)