from util.data import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号

def get_full_err_scores(test_result, val_result):
    # np_test_result = np.array(test_result) #3 2034 27
    np_test_result_pred = np.array(test_result[0])
    # np_test_result_pred[2072:2810] = np_test_result_pred[2072:2810]*2
    dim = np_test_result_pred.shape[1]

    np_test_result_true = np.array(test_result[1])
    np_test_result_label = np.array(test_result[2])
    np_test_result = [np_test_result_pred,np_test_result_true,np_test_result_label]
    # np_val_result = np.array(val_result) #3 31 27

    all_scores =  None
    all_normals = None
    feature_num = np_test_result[0].shape[1]

    # labels = np_test_result[2, :, 0].tolist()

    for i in range(feature_num):
        test_re_list = [np_test_result[0][:,i],np_test_result[1][:,i]]
        # val_re_list = np_val_result[:2,:,i]

        smoothed_err_scores, err_scores = get_err_scores(test_re_list, test_re_list)
        smoothed_normal_dist, normal_dist = get_err_scores(test_re_list, test_re_list)

        if all_scores is None:
            all_scores = err_scores#smoothed_err_scores
            all_normals = smoothed_normal_dist
        else:
            all_scores = np.vstack((
                all_scores,
                err_scores#smoothed_err_scores
            ))
            all_normals = np.vstack((
                all_normals,
                smoothed_normal_dist
            ))
    if dim == 1:
        all_scores = all_scores[np.newaxis, :]
        all_normals = all_normals[np.newaxis, :]

    all_scores = np.sum(all_scores,axis = 0)
    all_scores = all_scores[np.newaxis,:]
    all_normals  = np.sum(all_normals,axis = 0)
    all_normals = all_normals[np.newaxis, :]
    return all_scores, all_normals #27 2034    27 31


def get_final_err_scores(test_result, val_result):
    full_scores, all_normals = get_full_err_scores(test_result, val_result, return_normal_scores=True)

    all_scores = np.max(full_scores, axis=0)

    return all_scores



def get_err_scores(test_res, val_res):
    test_predict, test_gt = test_res
    val_predict, val_gt = val_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = np.abs(np.subtract(
                        np.array(test_predict).astype(np.float64), 
                        np.array(test_gt).astype(np.float64)
                    ))
    epsilon=1e-2

    err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) +epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)): #3 2034
        temp = err_scores[i-before_num:i+1]
        smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])

    
    return smoothed_err_scores, err_scores



def get_loss(predict, gt):
    return eval_mseloss(predict, gt)

def get_f1_scores(total_err_scores, gt_labels, topk=1):
    print('total_err_scores', total_err_scores.shape)
    # remove the highest and lowest score at each timestep
    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    
    topk_indices = np.transpose(topk_indices)

    total_topk_err_scores = []
    topk_err_score_map=[]
    # topk_anomaly_sensors = []

    for i, indexs in enumerate(topk_indices):
       
        sum_score = sum( score for k, score in enumerate(sorted([total_err_scores[index, i] for j, index in enumerate(indexs)])) )

        total_topk_err_scores.append(sum_score)

    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)

    return final_topk_fmeas

def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]

    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    thresold = np.max(normal_scores)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    gt_labels = np.array(gt_labels).squeeze()
    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)


    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return f1, pre, rec, auc_score, thresold





def get_best_performance_data(total_err_scores, gt_labels, topk=1):

    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
                                                        #    27 - 1 -1=     25  -    27                    [-1:]
    total_topk_err_scores = []
    topk_err_score_map=[]

    temp = np.take_along_axis(total_err_scores, topk_indices, axis=0)
    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    # 找出了每个时间点 误差最大的轴
    # plt.hist(total_topk_err_scores, bins=100, density=True, facecolor="blue", edgecolor="black", alpha=0.7)
    # # 显示横轴标签
    # plt.xlabel("区间")
    # # 显示纵轴标签
    # plt.ylabel("频数/频率")
    # # 显示图标题
    # plt.title("频数/频率分布直方图")
    # plt.show()


    final_topk_fmeas ,thresolds = eval_scores(total_topk_err_scores, gt_labels, 500, return_thresold=True)
    # 把2043分成 400份， 然后按照排序，分别将 > 份数的点当做outlier，然后算f1
    temp = max(final_topk_fmeas)
    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    thresold = thresolds[th_i]
    print(thresold)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1
    # folder_path = './results/'
    # dataset = 'power_demand'
    # np.savetxt(f'{folder_path}/{dataset}/bestF1_BiSeqMask_pred_label.csv',
    #            pred_labels, delimiter=",")
    gt_labels = np.array(gt_labels).squeeze()
    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return max(final_topk_fmeas), pre, rec, auc_score, thresold, pred_labels

