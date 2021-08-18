from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores
import os
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import f1_score, mean_squared_error
from tslearn.metrics import dtw
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch', help='batch size', type = int, default=16)
parser.add_argument('--epoch', help='train epoch', type = int, default=100)
parser.add_argument('--learning_rate', help='lr', type = float, default=0.001)

parser.add_argument('--slide_win', help='slide_win', type = int, default=168)

parser.add_argument('--slide_stride', help='slide_stride', type = int, default=16)
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

parser.add_argument('--hidden-size', default=3, type=float, help='hidden channel of module')
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
parser.add_argument('--kernel', default=3, type=int, help='kernel size')
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--save_path', type=str, default='checkpoint')

parser.add_argument('--pred_len', type=int, default=128)
parser.add_argument('--seq_mask_range_low', type=int, default=8)
parser.add_argument('--seq_mask_range_high', type=int, default=4)

parser.add_argument('--ensemble', type=int, default=0)
parser.add_argument('--model_type', type=str, default='BiPointMask')
parser.add_argument('--model_name', type=str, default='SCINet')
parser.add_argument('--point_part', type=int, default=12)

parser.add_argument('--lradj', type=int, default=3,help='adjust learning rate')

args = parser.parse_args()


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
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
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


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label)) #f1, precision, recall, TP, TN, FP, FN
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)



def get_dtw(test_result, val_result, interval = 100):
    pred = np.array(test_result[0])
    true = np.array(test_result[1])
    label = np.array(test_result[2])

    smoothed_pred = []
    for i in range(pred.shape[1]):
        smoothed = smooth(pred[:, i], window_len=5)
        smoothed_pred.append(smoothed[4::])
    smoothed_pred = np.stack(smoothed_pred,axis=1)
    rang = range(0,  pred.shape[0]-interval,interval)
    dtw_error =[]
    for j in rang:
        ps = smoothed_pred[j:j+interval]
        gt = true[j:j+interval]
        dist_ts = dtw(ps, gt)
        print(dist_ts)
        dtw_error.append(dist_ts)
    min_error = min(dtw_error)
    max_error = max(dtw_error)

    fineture_range= np.arange(min_error, max_error,0.1)
    best_f1 = []
    Finetunelabel = []
    for K in fineture_range:
        finetunelabel = np.zeros(label.shape[0])
        sc = dtw_error>K
        for k in range(len(dtw_error)):
            if sc[k] > 0:
                for kk in range(rang[k],rang[k]+interval,1):
                    finetunelabel[kk] = 1

        best_f1.append(f1_score(finetunelabel, label))
        Finetunelabel.append(finetunelabel)

    best_f1_final = np.max(best_f1)
    best_f1_final_index = np.where(best_f1==np.max(best_f1))
    print(f'Init F1 score: {best_f1_final}')

    pred_labels = Finetunelabel[int(best_f1_final_index[0])]

    # max(final_topk_fmeas), pre, rec, auc_score, thresold, pred_labels
    predict = adjust_predicts(score=label, label=label,
                              threshold=0,
                              pred=pred_labels,
                              calc_latency=False)

    f1, precision, recall, TP, TN, FP, FN = calc_point2point(predict, label)
    print(f'Seq F1 score: {f1}')
    print(f'Seq precision: {precision}')
    print(f'Seq recall: {recall}\n')

    plt.plot(rang, dtw_error)
    plt.plot(label*5)
    plt.plot(pred_labels * 6)
    plt.show()

    a = 0







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

    # folder_path = './checkpoint/'
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    # print(folder_path)
    # np.savetxt(
    #     f'{folder_path}/BiSeqMaskSwat/bestF1_{args.dataset}_{args.model_type}2Stack_lradj{args.lradj}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_pred_label.csv',
    #     top1_best_info[-1], delimiter=",")
    pred_labels = top1_best_info[5]
    thresold = top1_best_info[4]
    # max(final_topk_fmeas), pre, rec, auc_score, thresold, pred_labels
    predict = adjust_predicts(score=test_labels, label = test_labels,
                    threshold=thresold,
                    pred=pred_labels,
                    calc_latency=False)

    folder_path = './checkpoint/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print(folder_path)
    np.savetxt(
        f'{folder_path}/BiSeqMaskSwat/bestF1_Seq_{args.dataset}_{args.model_type}2Stack_lradj{args.lradj}_{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_pred_label.csv',
        predict, delimiter=",")

    f1, precision, recall, TP, TN, FP, FN = calc_point2point(predict, test_labels)

    print(f'F1 score: {info[0]}')
    print(f'precision: {info[1]}')
    print(f'recall: {info[2]}\n')

    print(f'Seq F1 score: {f1}')
    print(f'Seq precision: {precision}')
    print(f'Seq recall: {recall}\n')



if __name__ == "__main__":


    dim = 1
    if dim == 40:
        args.slide_win = 168
        args.slide_stride = 8
        args.hidden_size = 8
        args.batch = 32
        args.pred_len = 48
        args.lradj = 3
        args.dataset = 'SWAT'
        args.model_type = 'BiSeqMask'

        folder_path = './checkpoint/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print(folder_path)

        t_test_ground_list = np.loadtxt(
            f'{folder_path}/BiSeqMaskSwat/test_{args.dataset}_{args.model_type}2Stack_lradj{args.lradj}_'
            f'{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_ground_list.csv',delimiter=",")
        t_test_predicted_list = np.loadtxt(
            f'{folder_path}/BiSeqMaskSwat/test_{args.dataset}_{args.model_type}2Stack_lradj{args.lradj}_'
            f'{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_predicted_list.csv',delimiter=",")
        t_test_labels_list = np.loadtxt(
            f'{folder_path}/BiSeqMaskSwat/test_{args.dataset}_{args.model_type}2Stack_lradj{args.lradj}_'
            f'{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_labels_list.csv',delimiter=",")

        # t_test_labels_list[11590:11770] = 1
        # t_test_labels_list[38960:39000] = 1
        # t_test_labels_list[43660:43890] = 1
        # t_test_labels_list[100:1610] =1
        # t_test_labels_list[44350: 44520] = 0

        test_predicted_list = t_test_predicted_list.tolist()
        test_ground_list = t_test_ground_list.tolist()
        test_labels_list = t_test_labels_list.tolist()
        test_result = [test_predicted_list, test_ground_list, test_labels_list]

        get_dtw(test_result, test_result, interval = int(168/6))

        get_score(test_result, test_result)

    else:
        args.slide_win = 128
        args.slide_stride = 8
        args.hidden_size = 16.0
        args.batch = 32
        args.pred_len = 32
        args.lradj = 3
        args.dataset = 'SWAT'
        args.model_type = 'BiSeqMask'

        folder_path = './checkpoint/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print(folder_path)

        t_test_ground_list = np.loadtxt(
            f'{folder_path}/BiEvenSeqMaskSwat/1dim/test_{args.dataset}_{args.model_type}2Stack_1dim_group1_lradj{args.lradj}_'
            f'{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_ground_list.csv', delimiter=",")
        t_test_predicted_list = np.loadtxt(
            f'{folder_path}/BiEvenSeqMaskSwat/1dim/test_{args.dataset}_{args.model_type}2Stack_1dim_group1_lradj{args.lradj}_'
            f'{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_predicted_list.csv', delimiter=",")
        t_test_labels_list = np.loadtxt(
            f'{folder_path}/BiEvenSeqMaskSwat/1dim/test_{args.dataset}_{args.model_type}2Stack_1dim_group1_lradj{args.lradj}_'
            f'{args.slide_win}_h{args.hidden_size}_bt{args.batch}_p{args.pred_len}_labels_list.csv', delimiter=",")

        # t_test_labels_list[11590:11770] = 1
        # t_test_labels_list[38960:39000] = 1
        # t_test_labels_list[43660:43890] = 1
        # t_test_labels_list[100:1610] =1
        # t_test_labels_list[44350: 44520] = 0

        test_predicted_list = t_test_predicted_list[:,np.newaxis].tolist()
        test_ground_list = t_test_ground_list[:,np.newaxis].tolist()
        test_labels_list = t_test_labels_list[:,np.newaxis].tolist()
        test_result = [test_predicted_list, test_ground_list, test_labels_list]

        get_dtw(test_result, test_result, interval=int(128 / 4))

        get_score(test_result, test_result)