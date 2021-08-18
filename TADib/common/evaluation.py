import os
import sys
import copy
import json
import glob
import hashlib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from common.spot import SPOT
from common.utils import pprint


def evaluate_all(
    anomaly_score, anomaly_label, train_anomaly_score=None, q=1e-3, level=None
):
    # normalize anomaly_score
    print("Normalizing anomaly scores.")
    anomaly_score = normalize_1d(anomaly_score)
    if train_anomaly_score is not None:
        train_anomaly_score = normalize_1d(train_anomaly_score)

    # compute auc
    try:
        auc = roc_auc_score(anomaly_label, anomaly_score)
    except ValueError as e:
        auc = 0
        print("All zero in anomaly label, set auc=0")

    # iterate thresholds
    print("Iterating thresholds.")
    _, theta_iter, _, pred = iter_thresholds(
        anomaly_score, anomaly_label, metric="f1", normalized=True
    )

    _, adjust_pred = point_adjustment(pred, anomaly_label)

    metrics_iter = compute_point2point(pred, adjust_pred, anomaly_label)
    metrics_iter["delay"] = compute_delay(anomaly_label, pred)
    metrics_iter["auc"] = auc

    print("The following metrics are computed based on threshold iteration:")
    print(f"Threshold: {theta_iter:.3f}")
    pprint(metrics_iter)

    # EVT needs anomaly scores on training data for initialization
    if train_anomaly_score is not None:
        print("Finding thresholds via EVT.")
        theta_evt, pred_evt = compute_th_evt(
            train_anomaly_score, anomaly_score, anomaly_label, q, level
        )
        _, adjust_pred_evt = point_adjustment(pred, anomaly_label)
        metrics_evt = compute_point2point(pred_evt, adjust_pred_evt, anomaly_label)
        metrics_evt["delay"] = compute_delay(anomaly_label, pred_evt)
        metrics_evt["auc"] = auc
        print("\n" + "-" * 20 + "\n")
        print("The following metrics are computed based on threshold iteration:")
        print(f"Threshold: {theta_evt:.3f}")
        pprint(metrics_evt)

        return metrics_iter, metrics_evt, theta_iter, theta_evt
    return metrics_iter, theta_iter


def compute_point2point(pred, adjust_pred, label):
    return {
        "f1": f1_score(label, pred),
        "precision": precision_score(label, pred),
        "recall": recall_score(label, pred),
        "adj_f1": f1_score(label, adjust_pred),
        "adj_precision": precision_score(label, adjust_pred),
        "adj_recall": recall_score(label, adjust_pred),
    }


def compute_th_evt(train_score, test_score, label, q, level):
    results = []
    level_range = (
        [level]
        if level is not None
        else [0.001, 0.003, 0.005, 0.01, 0.1, 0.07, 0.0001, 0.00005]
    )
    for trial in ["higher", "lower"]:
        s = SPOT(q=q)  # SPOT object
        s.fit(train_score, test_score)  # data import
        for level in level_range:
            try:
                s.initialize(
                    level=level, min_extrema=(trial == "lower")
                )  # initialization step
                break
            except:
                pass
        ret = s.run(dynamic=False)  # run
        ths = np.nan_to_num(np.array(ret["thresholds"]))
        evt_th = np.mean(ths[ths <= 1])
        pred = (test_score <= evt_th).astype(int)
        f1 = f1_score(label, pred)
        results.append([f1, evt_th, pred])

    return max(results, key=lambda x: x[0])[1:]


metric_func = {
    "f1": f1_score,
    "pc": precision_score,
    "rc": recall_score,
    "auc": roc_auc_score,
}


def normalize_1d(arr):
    est = MinMaxScaler()
    return est.fit_transform(arr.reshape(-1, 1)).reshape(-1)


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(
            obj,
            fw,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


def store_benchmarking_results(
    hash_id,
    benchmark_dir,
    dataset,
    subdataset,
    args,
    model_name,
    anomaly_score,
    anomaly_label,
    time_tracker,
):
    value_store_dir = os.path.join(
        benchmark_dir, model_name, hash_id, dataset, subdataset
    )
    os.makedirs(value_store_dir, exist_ok=True)
    np.savez(os.path.join(value_store_dir, "anomaly_score"), anomaly_score)
    np.savez(os.path.join(value_store_dir, "anomaly_label"), anomaly_label)

    json_pretty_dump(time_tracker, os.path.join(value_store_dir, "time.json"))

    param_store_dir = os.path.join(benchmark_dir, model_name, hash_id)

    param_store = {"cmd": "python {}".format(" ".join(sys.argv))}
    param_store.update(args)

    json_pretty_dump(param_store, os.path.join(param_store_dir, "params.json"))
    print("Store output of {} to {} done.".format(model_name, param_store_dir))
    return os.path.join(benchmark_dir, model_name, hash_id, dataset)


class evaluator:
    def __init__(
        self,
        metrics,
        score,
        label,
        threshold=None,
        iterate_threshold=True,
        iterate_metric="f1",
        point_adjustment=True,
    ):
        assert (
            iterate_threshold and iterate_metric
        ), "[iterate_threshold] and [iterate_metric] should both be non-empty."

        assert (
            threshold is not None or iterate_threshold is not None
        ), "At least one of [threshold] and [iterate_threshold] should be non-empty."

        self.metrics = metrics
        self.score = score
        self.label = label
        self.pred = None
        self.pred_raw = None
        self.threshold = threshold
        self.__iterate_threshold = iterate_threshold
        self.__iterate_metric = iterate_metric
        self.__point_adjustment = point_adjustment

    def compute_pred(self):
        if self.threshold is not None:
            self.pred_raw = self.pred = (self.score >= self.threshold).astype(int)
        if self.__iterate_threshold:
            print("Iterating threshold.")
            best_metric, best_theta, best_adjust, best_raw = iter_thresholds(
                self.score,
                self.label,
                self.__iterate_metric,
                adjustment=self.__point_adjustment,
            )
            self.pred = best_adjust
            self.pred_raw = best_raw
        return self.pred

    def compute_metrics(self):
        if self.pred is None:
            self.compute_pred()
        results = {}
        for metric in self.metrics:
            if metric == "auc":
                results[metric] = metric_func[metric](self.label, self.score)
            else:
                results[metric] = metric_func[metric](self.pred, self.label)
                results["raw_" + metric] = metric_func[metric](
                    self.pred_raw, self.label
                )

        return results


def iter_thresholds(
    score, label, metric="f1", adjustment=False, normalized=False, threshold=None
):
    best_metric = -float("inf")
    best_theta = None
    best_adjust = None
    best_raw = None
    adjusted_pred = None
    if threshold is not None:
        search_range = [0]
    else:
        search_range = np.linspace(0, 1, 100)

    best_set = []
    for trial in ["higher", "less"]:
        for anomaly_ratio in search_range:
            if threshold is None:
                if not normalized:
                    theta = np.percentile(score, 100 * (1 - anomaly_ratio))
                else:
                    theta = anomaly_ratio
            else:
                theta = threshold
            if trial == "higher":
                pred = (score >= theta).astype(int)
            elif trial == "less":
                pred = (score <= theta).astype(int)

            if adjustment:
                pred, adjusted_pred = point_adjustment(pred, label)
            else:
                adjusted_pred = pred

            current_value = metric_func[metric](adjusted_pred, label)
            # print(anomaly_ratio, current_value)

            if current_value > best_metric:
                best_metric = current_value
                best_adjust = adjusted_pred
                best_raw = pred
                best_theta = theta
        best_set.append((best_metric, best_theta, best_adjust, best_raw))

    return max(best_set, key=lambda x: x[0])
    # return best_metric, best_theta, best_adjust, best_raw


def point_adjustment(pred, label):
    """
    Borrow from https://github.com/NetManAIOps/OmniAnomaly/blob/master/omni_anomaly/eval_methods.py
    """
    adjusted_pred = copy.deepcopy(pred)

    anomaly_state = False
    anomaly_count = 0
    latency = 0
    for i in range(len(adjusted_pred)):
        if label[i] and adjusted_pred[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not label[j]:
                    break
                else:
                    if not adjusted_pred[j]:
                        adjusted_pred[j] = True
                        latency += 1
        elif not label[i]:
            anomaly_state = False
        if anomaly_state:
            adjusted_pred[i] = True
    return pred, adjusted_pred


def compute_support(score, label, dtype="normal"):
    if dtype == "normal":
        score_idx = np.arange(len(score)).reshape(-1, 1)[(label == 0).astype(bool)]
    elif dtype == "anomaly":
        score_idx = np.arange(len(score)).reshape(-1, 1)[(label == 1).astype(bool)]

    clusters = []
    dscore = score[score_idx]
    clustering = AgglomerativeClustering(affinity="l1", linkage="complete").fit(dscore)
    cluster_labels = clustering.labels_

    for label in range(len(set(cluster_labels))):
        clusters.append(dscore[cluster_labels == label])
    max_label = max(enumerate(clusters), key=lambda x: np.mean(x[1]))[0]

    max_cluster = clusters[max_label]
    std = np.std(max_cluster)
    mean = np.mean(max_cluster)
    original_idx = score_idx[cluster_labels == max_label]

    ## plot internal
    #     plot_x = np.arange(len(dscore))
    #     scatter_x = plot_x[cluster_labels==max_label]
    #     plt.figure()
    #     plt.plot(plot_x, dscore)
    #     plt.scatter(scatter_x, max_cluster, c="r")
    #     plt.hlines(mean, 0, cluster_labels.shape[0], "r", label=f"mean:{mean:.3f}")
    #     mean_arr = np.array([mean] * len(plot_x))
    #     plt.fill_between(plot_x, mean_arr-std, mean_arr+std, alpha=0.2, facecolor = "green")
    #     plt.show()

    return_dict = {"mean": mean, "std": std, "idx": original_idx}
    return return_dict


def compute_salience(score, label, plot=False, ax=None, fig_saving_path=""):
    # print("Computing salience")
    score = score.reshape(-1, 1)
    label = label.reshape(-1, 1)
    total_indice = np.arange(len(score)).reshape(-1, 1)
    score_n = score[~label.astype(bool)]
    score_a = score[label.astype(bool)]

    # score_n_idx = total_indice[~label.astype(bool)]
    n_dict = compute_support(score, label, "normal")
    salient_score_n = score[n_dict["idx"]]

    score_a_idx = total_indice[label.astype(bool)]
    a_dict = compute_support(score, label, "anomaly")
    salient_score_a = score[a_dict["idx"]]

    a_lower = a_dict["mean"] - a_dict["std"]
    n_upper = n_dict["mean"] + n_dict["std"]

    # print(n_dict["mean"] - n_dict["std"], n_dict["mean"] + n_dict["std"])
    # print(a_dict["mean"] - a_dict["std"], a_dict["mean"] + a_dict["std"])

    overlapping = (n_upper - a_lower if n_upper >= a_lower else 0) / (
        2 * (min(a_dict["std"], n_dict["std"]))
    )
    non_overlapping = 1 - overlapping
    a_count_ratio = len(a_dict["idx"]) / (len(a_dict["idx"]) + len(n_dict["idx"]))
    salience = non_overlapping * (
        a_count_ratio * a_dict["mean"] - (1 - a_count_ratio) * n_dict["mean"]
    )

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(score, c="b", label="score")
        ax.plot(label, c="g", label="label")
        ax.hlines(
            n_dict["mean"],
            0,
            label.shape[0],
            "b",
            label=f"normal_plane:{n_dict['mean']:.3f}",
        )
        ax.hlines(
            a_dict["mean"],
            0,
            label.shape[0],
            "r",
            label=f"anomaly_plane:{a_dict['mean']:.3f}",
        )
        ax.hlines(0, 0, label.shape[0], "r", label=f"salience:{salience:.3f}")
        ax.hlines(0, 0, label.shape[0], "r", label=f"overlapping:{overlapping:.3f}")
        ax.scatter(n_dict["idx"], salient_score_n, c="g")
        ax.scatter(a_dict["idx"], salient_score_a, c="r")

        ax.fill_between(
            np.arange(len(score)),
            a_dict["mean"] - a_dict["std"],
            a_dict["mean"] + a_dict["std"],
            alpha=0.2,
            facecolor="red",
        )
        ax.fill_between(
            np.arange(len(score)),
            n_dict["mean"] - n_dict["std"],
            n_dict["mean"] + n_dict["std"],
            alpha=0.2,
            facecolor="green",
        )
        ax.legend()

        if fig_saving_path:
            ax.figure.savefig(fig_saving_path)

    salience = 1 / (1 + np.exp(-1 * salience))

    return salience


def evaluate_benchmarking_folder(
    folder, benchmarking_dir, hash_id, dataset, model_name
):
    total_adj_f1 = []
    total_train_time = []
    total_test_time = []
    folder_count = 0
    for folder in glob.glob(os.path.join(folder, "*")):
        folder_name = os.path.basename(folder)
        print("Evaluating {}".format(folder_name))
        anomaly_score = np.load(
            os.path.join(folder, "anomaly_score.npz"), allow_pickle=True
        )["arr_0"].item()["test"]

        anomaly_label = np.load(os.path.join(folder, "anomaly_label.npz"))[
            "arr_0"
        ].astype(int)
        with open(os.path.join(folder, "time.json")) as fr:
            time = json.load(fr)

        best_f1, best_theta, best_adjust_pred, best_raw_pred = iter_thresholds(
            anomaly_score, anomaly_label, metric="f1", adjustment=True
        )

        try:
            auc = roc_auc_score(anomaly_label, anomaly_score)
        except ValueError as e:
            auc = 0
            print("All zero in anomaly label, set auc=0")

        metrics_iter, theta_iter = evaluate_all(anomaly_score, anomaly_label)

        total_adj_f1.append(metrics_iter["adj_f1"])
        total_train_time.append(time["train"])
        total_test_time.append(time["test"])

        metrics_iter["train"] = time["train"]
        metrics_iter["test"] = time["test"]
        json_pretty_dump(metrics_iter, os.path.join(folder, "metrics.json"))
        folder_count += 1

    total_adj_f1 = np.array(total_adj_f1)
    adj_f1_mean = total_adj_f1.mean()
    adj_f1_std = total_adj_f1.std()

    train_time_sum = sum(total_train_time)
    test_time_sum = sum(total_test_time)

    with open(
        os.path.join(benchmarking_dir, f"{dataset}_{model_name}.txt"), "a+"
    ) as fw:
        params = " ".join(sys.argv)
        info = f"{hash_id}\tcount:{folder_count}\t{params}\ttrain:{train_time_sum:.4f} test:{test_time_sum:.4f}\tadj f1: [{adj_f1_mean:.4f}({adj_f1_std:.4f})]\n"
        fw.write(info)
    print(info)


def compute_delay(label, pred):
    def onehot2interval(arr):
        result = []
        record = False
        for idx, item in enumerate(arr):
            if item == 1 and not record:
                start = idx
                record = True
            if item == 0 and record:
                end = idx  # not include the end point, like [a,b)
                record = False
                result.append((start, end))
        return result

    count = 0
    total_delay = 0
    pred = np.array(pred)
    label = np.array(label)
    for start, end in onehot2interval(label):
        pred_interval = pred[start:end]
        if pred_interval.sum() > 0:
            total_delay += np.where(pred_interval == 1)[0][0]
            count += 1
    return total_delay


if __name__ == "__main__":
    anomaly_label = [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1]
    anomaly_score = np.random.uniform(0, 1, size=len(anomaly_label))
    evaluate_all(anomaly_score, anomaly_label)