import numpy as np
import pandas as pd
from typing import Any, Tuple, Callable, Dict, List
from utils.DataClass import DataClass
from utils.plotting_utils import plot_label_distribution, plot_probas_distribution

import os

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

import matplotlib.pyplot as plt

from dataclasses import dataclass


def kl_divergence(gt: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    notation:
    p - corresponds to gt
    q - corresponds to our prediction
    """
    p = get_label_frequencies(gt)
    q = get_label_frequencies(pred_labels)

    kl_div = np.mean(np.where((p != 0) & (q != 0), p * np.log(p / q), 0))
    return kl_div


def avg_size_of_pred_set(y: np.ndarray) -> float:
    return np.mean(np.sum(y, axis=1))


@dataclass
class MetricResult:
    mean: float
    std: float

    @classmethod
    def mean_std(cls, list) -> "MetricResult":
        return cls(np.mean(list, axis=0), np.std(list, axis=0))

    @classmethod
    def eval(cls, func, list_of_args) -> "MetricResult":
        res = [func(arg) for arg in list_of_args]

        return cls.mean_std(res)


def get_label_frequencies(y: np.ndarray) -> np.ndarray:
    return y.sum(axis=0) / y.shape[0]


class Metric:
    def __init__(
        self,
        metric_func: Callable[[np.ndarray, np.ndarray], float],
        required_arr: str,
        name: str = None,
        **kwargs,
    ) -> None:
        self.metric_func = metric_func
        self.required_arr = required_arr
        self.kwargs = kwargs

        if name is None:
            if len(kwargs) == 0:
                self.__name__ = metric_func.__name__
            else:
                self.__name__ = f"{metric_func.__name__}_{kwargs}"

        else:
            self.__name__ = name

    def __call__(self, gt, pred_labels, probas) -> float:
        if self.required_arr == "probas":
            values = probas
        elif self.required_arr == "pred_labels":
            values = pred_labels
        return self.metric_func(gt, values, **self.kwargs)

    def eval(
        self,
        gt_list: list[np.ndarray],
        pred_labels: list[np.ndarray],
        probas: list[np.ndarray],
    ) -> MetricResult:
        metrics = []
        for i in range(len(gt_list)):
            value = self(gt_list[i], pred_labels[i], probas[i])
            # print(value)
            metrics.append(value)

        results = MetricResult.mean_std(metrics)
        
        return results


class ExperimentInfo:
    class LabelInfo:
        def __init__(
            self,
            gt: list[np.ndarray],
            scores: list[np.ndarray],
            probas: list[np.ndarray],
            pred_labels: list[np.ndarray],
        ) -> None:
            """
            Note: args scores, probas provided for class interface consistency and fo future usage
            This proxy class incapsulates per label analysis and operations.
            """
            freqs_gt = get_label_frequencies(gt[0])
            sorted_indices = np.argsort(freqs_gt)[::-1]
            # self.sorted_indices = sorted_indices
            self.num_of_labels = gt[0].shape[1]

            # freqs_pred = MetricResult.eval(get_label_frequencies, pred_labels)

            # df = pd.DataFrame()

            rows_for_df = []
            for i in sorted_indices:
                # Calculate metrics for each run and label
                precision_metrics = [
                    precision_score(
                        gt[run_index][:, i],
                        pred_labels[run_index][:, i],
                        zero_division=0,
                    )
                    for run_index in range(len(gt))
                ]
                recall_metrics = [
                    recall_score(
                        gt[run_index][:, i],
                        pred_labels[run_index][:, i],
                        zero_division=0,
                    )
                    for run_index in range(len(gt))
                ]
                accuracy_metrics = [
                    accuracy_score(gt[run_index][:, i], pred_labels[run_index][:, i])
                    for run_index in range(len(gt))
                ]

                # Aggregate metrics
                precision_result = MetricResult.mean_std(precision_metrics)
                recall_result = MetricResult.mean_std(recall_metrics)
                accuracy_result = MetricResult.mean_std(accuracy_metrics)

                # Construct new row with mean and std for metrics
                new_row = {
                    "Label": i,
                    "Frequency of the label": freqs_gt[i],
                    "Prediction label frequency (Mean)": MetricResult.eval(
                        get_label_frequencies, [pred[:, i] for pred in pred_labels]
                    ).mean,
                    "Prediction label frequency (Std)": MetricResult.eval(
                        get_label_frequencies, [pred[:, i] for pred in pred_labels]
                    ).std,
                    "Precision (Mean)": precision_result.mean,
                    "Precision (Std)": precision_result.std,
                    "Recall (Mean)": recall_result.mean,
                    "Recall (Std)": recall_result.std,
                    "Accuracy (Mean)": accuracy_result.mean,
                    "Accuracy (Std)": accuracy_result.std,
                }
                rows_for_df.append(new_row)
                # df = df.append(new_row, ignore_index=True)

            self.df = pd.DataFrame(rows_for_df)
            self.freqs = freqs_gt[sorted_indices]
            self.sorted_indices = sorted_indices
            # Reorder arrays based on sorted indices
            self.gt = [g[:, sorted_indices] for g in gt]
            self.scores = [s[:, sorted_indices] for s in scores]
            self.probas = [p[:, sorted_indices] for p in probas]
            self.pred_labels = [pred[:, sorted_indices] for pred in pred_labels]

        def metric_per_label(self, metric: Metric) -> list:
            """Compute metric across all labels and runs, returning mean and std."""

            # metric_values = [
            #     metric(self.gt[:, i], self.pred_labels[:, i], self.probas[:, i])
            #     for i in range(self.num_of_labels)
            # ]

            metric_values = []
            for i in range(self.num_of_labels):
                # for i in range(len(self.gt)):
                vals = [
                    metric(
                        self.gt[j][:, i],
                        self.pred_labels[j][:, i],
                        self.probas[j][:, i],
                    )
                    for j in range(len(self.gt))
                ]
                metric_values.append(MetricResult.mean_std(vals))

                # metric_values = MetricResult.eval(lambda x: metric_func(x[0][:, i], x[1][:, i]),
                #                                   list(zip(self.gt, self.pred_labels)))
                # result.append(metric_values)

            return metric_values

    class SetInfo:
        """
        aggregated statistics per sets:
        - metric per set size
        - distribution of set sizes
        - distriubtion of set sizes differences
        """

        def __init__(
            self,
            gt: list[np.ndarray],
            scores: list[np.ndarray],
            probas: list[np.ndarray],
            pred_labels: list[np.ndarray],
        ) -> None:
            self.gt, self.scores, self.probas, self.pred_labels = (
                gt,
                scores,
                probas,
                pred_labels,
            )

            self.gt_set_sizes = [np.sum(gt_run, axis=1).astype(int) for gt_run in gt]
            self.pred_set_sizes = [np.sum(pred_labels_run, axis=1).astype(int) for pred_labels_run in pred_labels]

            self.max_set_size = np.max(self.gt_set_sizes[0]).astype(int)

            # because of the filtration we don't have set with 0 size in gt
            self.set_sizes = list(range(1, self.max_set_size + 1))
            self.set_sizes_distribution = np.bincount(self.gt_set_sizes[0])

        def metric_per_set_size(self, metric: Metric) -> list:
            """
            :param required_arr: is "probas" or "pred_labels"

            not all metrics and their averages are supported,
            because not all labels are present for e.g. in records with set size 1
            """

            metric_values = []
            sizes = []
            for size in self.set_sizes:
                # Aggregated metric results for each set size across runs
                size_metrics = []

                for run_index, gt_run in enumerate(self.gt):
                    indices = np.where(self.gt_set_sizes[run_index] == size)[0]
                    if len(indices) == 0:
                        continue  # Skip if no sets of this size in the current run

                    # Filter gt, pred_labels, and probas for the current set size
                    gt_filtered = gt_run[indices]
                    pred_labels_filtered = self.pred_labels[run_index][indices]
                    probas_filtered = self.probas[run_index][indices]

                    # Compute and collect the metric for the current run and set size
                    if len(gt_filtered) > 0 and len(pred_labels_filtered) > 0:
                        size_metrics.append(metric(gt_filtered, pred_labels_filtered, probas_filtered))

                # Compute mean and std of the collected metrics for the current set size across all runs
                if size_metrics:
                    metric_values.append(MetricResult.mean_std(size_metrics))
                    sizes.append(size)
                # else: 
                    # metric_values.append(MetricResult(None, None))

            return metric_values, sizes

    def get_metrics(self, metric_list: List[Metric]) -> dict:
        """
        : param metric_list: is expected to be like for e.g. [(roc_auc_score, "probas", "weighted"), (precision_score, "pred_labels", "macro")]
        """
        result = {}
        for metric in metric_list:
            result[f"{metric.__name__}(mean)"] = metric.eval(
                self.gt, self.pred_labels, self.probas
            ).mean
            result[f"{metric.__name__}(std)"] = metric.eval(
                self.gt, self.pred_labels, self.probas
            ).std

        return result

    def __init__(self, data: DataClass) -> None:
        thresholds_list = data.get_thresholds(
            f1_score, "max", is_thresholds_independent=True, average="weighted"
        )
        (
            self.gt,
            self.scores,
            self.probas,
            self.pred_labels,
        ) = data.get_masked_and_thresholded(thresholds_list)

        self.model_name = data.model_name
        self.dataset_name = data.dataset_name

        self.label_info = ExperimentInfo.LabelInfo(
            self.gt, self.scores, self.probas, self.pred_labels
        )
        self.set_info = ExperimentInfo.SetInfo(
            self.gt, self.scores, self.probas, self.pred_labels
        )


class ModelComparison:
    def __init__(self, *args: DataClass, path: str = "./") -> None:
        """
        : param path: path where to create "data" directory with all plots and results. On default, directory is created in one where the script is invoked
        """

        self.experiments = [ExperimentInfo(data) for data in args]

        if self.check_data() == False:
            raise ValueError("There is an error with datasets, initialization aborted")
        else:
            print("All checks are succesful")

        some_experiment = self.experiments[0]

        self.dataset_name = some_experiment.dataset_name

        self.num_of_labels = some_experiment.label_info.num_of_labels
        self.freqs = some_experiment.label_info.freqs
        self.max_set_size = some_experiment.set_info.max_set_size

        self.plots_path = os.path.join(path, f"data/{self.dataset_name}/plots")
        self.data_path = os.path.join(path, f"data/{self.dataset_name}")

        os.makedirs(self.plots_path, exist_ok=True)

    def check_data(self) -> bool:
        is_ok = True

        # check whether we work on dataset with same name
        dataset_names = [exp.dataset_name for exp in self.experiments]
        if len(set(dataset_names)) != 1:
            print(
                "ERROR: Check the DataClass objects, \
                  it seems they are from different datasets and therefore we can't compare them"
            )
            print(dataset_names)
            is_ok = False
            # raise ValueError
        else:
            print("Experiments are performed on dataset with common name")

        # check that frequencies from gt are same
        # freq_arrs = [exp.label_info.freqs for exp in self.experiments]
        # # for exp in self.experiments:
        # #     print(exp.model_name)
        # #     print(exp.label_info.freqs.shape)
        # if not all(np.allclose(freq_arrs[0], freq_arr, rtol=0.01) for freq_arr in freq_arrs):
        #     print("ERROR: Seems that gt differs between experiments")
        #     for exp in self.experiments:
        #         print(exp.model_name)
        #         print(exp.label_info.freqs)
        #     is_ok = False
        #     # raise ValueError
        # else:
        #     print("Gt frequencies are equal between experiments")

        # check that max set sizes are same
        max_set_sizes = [exp.set_info.max_set_size for exp in self.experiments]
        if len(set(max_set_sizes)) != 1:
            print("ERROR: Max set sizes are unequal between datasets")
            print(max_set_sizes)
            is_ok = False
            # raise ValueError
        else:
            print("Max set sizes are equal between datasets")

        # # check whether the set sizes distributions are same
        # set_sizes_distributions = [
        #     exp.set_info.set_sizes_distribution for exp in self.experiments
        # ]
        # if not all(
        #     np.array_equal(set_sizes_distributions[0], set_sizes_distribution)
        #     for set_sizes_distribution in set_sizes_distributions
        # ):
        #     print("ERROR: Set sizes distributions in gt are not equal")
        #     is_ok = False
        #     # raise ValueError
        # else:
        #     print("Ground truth set sizes distributions are equal")

        return is_ok

    def plot_info(
        self,
        metric_list: List[Metric],
        metric_list_for_labels: List[Metric],
        figsize: Tuple[int, int] = (18, 10),
        save: bool = False,
        show: bool = True,
    ) -> None:
        """plots multiple canvases with different metrics"""
        for metric in metric_list_for_labels:
            self.plot_metric_per_label(metric, figsize=figsize, save=save, show=show)

        self.plot_label_distribution(figsize=figsize, save=save, show=show)

        for metric in metric_list:
            self.plot_metric_per_set_size(metric, save=save, show=show)

        self.plot_set_sizes_distribution(figsize=figsize, save=save, show=show)
        self.plot_set_sizes_differences_distribution(
            figsize=figsize, save=save, show=show
        )
        self.plot_per_basket_errors_distribution(figsize=figsize, save=save, show=show)

    def plot_label_distribution(
        self, figsize: Tuple[int, int] = (18, 10), save: bool = False, show: bool = True
    ) -> None:
        """this function supports plotting of distribution of multiple arrays"""
        # Calculate the number of labels
        
        
        # gt = self.experiments[0].gt
        num_labels = self.experiments[0].gt[0].shape[1]
        print(num_labels)
        
        for exp in self.experiments:
            gt = []
            for g in exp.gt:
                gt.append(g.mean(axis=0))
        
        gt_occ_mean = np.mean(gt, axis=0)
        gt_occ_std = np.std(gt, axis=0)
        
        print(gt_occ_mean)
        
        # Calculate label occurrences for each dataset
        occurrences = [gt_occ_mean]
        occurrences_std = [gt_occ_std]
        
        model_names = ["Ground Truth"]
        
        for exp in self.experiments:
            p_l = []
            for pred_labels in exp.pred_labels:
                p_l.append(pred_labels.mean(axis=0))
                
            occurrences.append(np.mean(p_l, axis=0))
            occurrences_std.append(np.std(p_l, axis=0)) 
            model_names.append(exp.model_name)
            
            
        # Determine the number of groups and bar width
        n_groups = num_labels
        n_labels = len(occurrences)
        bar_width = 1 / (n_labels + 1)
        # Create figure
        plt.figure(figsize=figsize)
        # Create an index for each group
        index = np.arange(n_groups)
        # plt.bar(index + 0 * bar_width, gt.sum(axis=0) / gt.shape[0], bar_width, label=f'Ground truth', alpha=0.5)

        # Plot each dataset
        for i, occ in enumerate(occurrences):
            plt.bar(
                index + i * bar_width,
                occ,
                bar_width,
                label=f"{model_names[i]}",
                alpha=0.5,
            )
            
            plt.errorbar(
                index + i * bar_width,  # Center error bars on bars
                occ,
                yerr=occurrences_std[i],  # Include standard deviation as error bars
                fmt="none",  # No marker for error bars
                color="black",  # Error bar color
                capsize=5,  # Cap size for error bars
                linewidth=1,  # Line width of error bars
            )

        plt.xlabel("Label Index")
        plt.ylabel("Frequency")
        plt.title("Distribution of Label Frequencies")
        plt.xticks(index + bar_width / 2 * (n_labels - 1), index)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save:
            plt.savefig(
                os.path.join(
                    self.plots_path,
                    f"Label distribution on dataset {self.dataset_name}.png",
                ),
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )

        if show:
            plt.show()

    def plot_metric_per_label(
        self,
        metric: Metric,
        figsize: Tuple[int, int] = (18, 10),
        save: bool = False,
        show: bool = True,
    ) -> None:
        """
        plots multiple distributions on one canvas with specified metric

        :param metric: A metric function that takes ground truth and predicted labels as input.
        :param required_arr: is "probas" or "pred_labels"
        """

        plt.figure(figsize=figsize)
        plt.grid(True)

        # Get the number of labels from the first experiment (assuming all have the same number)
        # num_labels = self.experiments[0].label_info.num_of_labels
        bar_width = 1 / (len(self.experiments) + 1)
        index = np.arange(self.num_of_labels)

        # Iterate over each experiment and plot the metric
        for i, exp in enumerate(self.experiments):
            metric_values = exp.label_info.metric_per_label(metric)
            metric_mean = [m.mean for m in metric_values]
            metric_std = [m.std for m in metric_values]
            
            
            plt.bar(
                index + i * bar_width,  # Adjust position to avoid overlap
                metric_mean,
                bar_width,
                label=exp.model_name,
                alpha=0.5,
            )
            
            plt.errorbar(
                index + i * bar_width,  # Center error bars on bars
                metric_mean,
                yerr=metric_std,  # Include standard deviation as error bars
                fmt="none",  # No marker for error bars
                color="black",  # Error bar color
                capsize=5,  # Cap size for error bars
                linewidth=1,  # Line width of error bars
            )

        # Setting the x-axis labels to show label frequencies
        label_frequencies = self.experiments[0].label_info.freqs
        plt.xticks(
            index + bar_width / 2 * (len(self.experiments) - 1), label_frequencies
        )

        plt.xlabel("Label Frequency")
        plt.ylabel(f"{metric.__name__} Value")
        plt.title(
            f"Distribution of {metric.__name__} per label across experiments on dataset {self.dataset_name}"
        )
        plt.legend()
        plt.tight_layout()

        if save:
            plt.savefig(
                os.path.join(self.plots_path, f"{metric.__name__}_per_label.png"),
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )

        if show:
            plt.show()

    def plot_set_sizes_distribution(
        self, figsize: Tuple[int, int] = (18, 10), save: bool = False, show: bool = True
    ) -> None:
        """
        Plots the distribution of set sizes for the ground truth and each dataset in self.experiments.

        :param figsize: Size of the plot.
        """

        plt.figure(figsize=figsize)
        plt.grid(True)

        # Find the maximum set size across all experiments
        max_set_size = max([exp.set_info.max_set_size for exp in self.experiments])

        # Create an index array
        index = np.arange(1, max_set_size + 1)  # +1 because set sizes start from 1

        # Determine bar width
        bar_width = 1 / (
            len(self.experiments) + 2
        )  # +2 for ground truth and extra spacing

        # Plotting the distribution of set sizes for ground truth
        
        for i, exp in enumerate(self.experiments):
            
            sz = []
            for set_sizes in exp.set_info.gt_set_sizes:
                sz.append([np.sum(set_sizes == j) for j in index])
        gt_set_sizes_distribution = np.mean(sz, axis=0)
        gt_set_sizes_std = np.std(sz, axis=0)
            
        
        # gt_set_sizes_distribution = [
        #     np.sum(self.experiments[0].set_info.gt_set_sizes[0] == i) for i in index
        # ]
        
        plt.bar(
            index - bar_width,
            gt_set_sizes_distribution,
            bar_width,
            label="Ground Truth",
            alpha=0.5,
            color="g",
        )
        
        plt.errorbar(
            index - bar_width,  # Center error bars on bars
            gt_set_sizes_distribution,
            yerr=gt_set_sizes_std,  # Include standard deviation as error bars
            fmt="none",  # No marker for error bars
            color="black",  # Error bar color
            capsize=5,  # Cap size for error bars
            linewidth=1,  # Line width of error bars
        )

        # Iterate over each experiment and plot the distribution of set sizes
        for i, exp in enumerate(self.experiments):
            
            sz = []
            for set_sizes in exp.set_info.pred_set_sizes:
                sz.append([np.sum(set_sizes == j) for j in index])
            
            pred_set_sizes_distribution = np.mean(sz, axis=0)
            pred_set_sizes_std = np.std(sz, axis=0)
            
            # pred_set_sizes_distribution = [
            #     np.sum(exp.set_info.pred_set_sizes == j) for j in index
            # ]
            plt.bar(
                index + i * bar_width,
                pred_set_sizes_distribution,
                bar_width,
                label=exp.model_name,
                alpha=0.5,
            )
            
            plt.errorbar(
                index + i * bar_width,  # Center error bars on bars
                pred_set_sizes_distribution,
                yerr=pred_set_sizes_std,  # Include standard deviation as error bars
                fmt="none",  # No marker for error bars
                color="black",  # Error bar color
                capsize=5,  # Cap size for error bars
                linewidth=1,  # Line width of error bars
            )

        # Setting the x-axis labels to show set sizes
        plt.xticks(index, index)

        plt.xlabel("Set size")
        plt.ylabel("Frequency")
        plt.title(
            f"Distribution of set sizes in ground truth and across experiments on dataset {self.dataset_name}"
        )
        plt.legend()
        plt.tight_layout()

        if save:
            plt.savefig(
                os.path.join(self.plots_path, f"set_sizes_distribution.png"),
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )

        if show:
            plt.show()

    def plot_set_sizes_differences_distribution(
        self, figsize: Tuple[int, int] = (18, 10), save: bool = False, show: bool = True
    ) -> None:
        """
        Plots the distribution of set size differences between predictions and ground truth for each dataset in self.experiments.

        :param figsize: Size of the plot.
        """

        plt.figure(figsize=figsize)
        plt.grid(True)

        # Find the maximum possible set size difference
        max_set_size_difference = max(
            [exp.set_info.max_set_size for exp in self.experiments]
        )

        # Create an index array for plotting
        index = np.arange(
            max_set_size_difference + 1
        )  # +1 to include the max difference

        # Determine bar width
        bar_width = 1 / (len(self.experiments) + 1)  # +1 for extra spacing

        # Iterate over each experiment and plot the distribution of set size differences
        for i, exp in enumerate(self.experiments):
            
            sz = []
            for j in range(len(exp.gt)):
            
                # Calculate set size differences
                set_size_diffs = np.abs(
                    np.sum(exp.gt[j], axis=1) - np.sum(exp.pred_labels[j], axis=1)
                )
                set_size_diffs_distribution = [np.sum(set_size_diffs == k) for k in index]
                sz.append(set_size_diffs_distribution)
                
            set_size_diffs_distribution = np.mean(sz, axis=0)
            set_size_diffs_std = np.std(sz, axis=0)

            # Plotting
            plt.bar(
                index + i * bar_width,
                set_size_diffs_distribution,
                bar_width,
                label=exp.model_name,
                alpha=0.5,
            )
            
                    
            plt.errorbar(
                index + i* bar_width,  # Center error bars on bars
                set_size_diffs_distribution,
                yerr=set_size_diffs_std,  # Include standard deviation as error bars
                fmt="none",  # No marker for error bars
                color="black",  # Error bar color
                capsize=5,  # Cap size for error bars
                linewidth=1,  # Line width of error bars
            )

        # Setting the x-axis labels to show set size differences
        plt.xticks(index, index)

        plt.xlabel("Set size difference")
        plt.ylabel("Frequency")
        plt.title("Distribution of set size differences across experiments")
        plt.legend()
        plt.tight_layout()

        if save:
            plt.gcf().set_facecolor("white")
            plt.savefig(
                os.path.join(self.plots_path, f"set_sizes_difference_distribution.png"),
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )

        if show:
            plt.show()

    def plot_per_basket_errors_distribution(
        self, figsize: Tuple[int, int] = (18, 10), save: bool = False, show: bool = True
    ) -> None:
        """
        Plots the distribution of the number of errors per instance for each dataset in self.experiments.

        :param figsize: Size of the plot.
        """

        plt.figure(figsize=figsize)
        plt.grid(True)

        # Find the maximum number of possible errors (which is the number of labels)
        max_errors = max([exp.label_info.num_of_labels for exp in self.experiments])

        # Create an index array for plotting
        index = np.arange(max_errors + 1)  # +1 to include the max number of errors

        # Determine bar width
        bar_width = 1 / (len(self.experiments) + 1)  # +1 for extra spacing

        # Iterate over each experiment and plot the distribution of errors per basket
        for i, exp in enumerate(self.experiments):
            # Calculate the number of errors per basket
            er = []
            for j in range(len(exp.gt)):                
                errors_per_basket = np.sum(np.abs(exp.gt[j] - exp.pred_labels[j]), axis=1)
                errors_distribution = [np.sum(errors_per_basket == k) for k in index]
                er.append(errors_distribution)
            
            errors_distribution = np.mean(er, axis=0)
            errors_std = np.std(er, axis=0)

            # Plotting
            plt.bar(
                index + i * bar_width,
                errors_distribution,
                bar_width,
                label=exp.model_name,
                alpha=0.5,
            )
            
            plt.errorbar(
                index + i* bar_width,  # Center error bars on bars
                errors_distribution,
                yerr=errors_std,  # Include standard deviation as error bars
                fmt="none",  # No marker for error bars
                color="black",  # Error bar color
                capsize=5,  # Cap size for error bars
                linewidth=1,  # Line width of error bars
            )

        # Setting the x-axis labels to show number of errors per basket
        plt.xticks(index, index)

        plt.xlabel("Number of errors per basket")
        plt.ylabel("Frequency")
        plt.title("Distribution of per-basket errors across experiments")
        plt.legend()
        plt.tight_layout()

        if save:
            # plt.gcf().set_facecolor('white')
            plt.savefig(
                os.path.join(self.plots_path, f"per_basket_errors_distribution.png"),
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )

        if show:
            plt.show()

    def plot_metric_per_set_size(
        self,
        metric: Metric,
        figsize: Tuple[int, int] = (18, 10),
        save: bool = False,
        show: bool = True,
    ) -> None:
        """
        Plots multiple distributions of a specified metric per set size on one canvas.

        :param metric: A metric function that takes ground truth and predicted labels as input.
        :param required_arr: is "probas" or "pred_labels"
        :param figsize: Size of the plot.
        """

        plt.figure(figsize=figsize)
        plt.grid(True)

        # Determine the number of set sizes and calculate the bar width
        num_set_sizes = self.max_set_size
        bar_width = 1 / (len(self.experiments) + 1)
        index = np.arange(num_set_sizes)

        # Iterate over each experiment and plot the metric for each set size
        for i, exp in enumerate(self.experiments):
            # try:
            metric_values, sizes = exp.set_info.metric_per_set_size(metric)
            metric_mean = [m.mean for m in metric_values]
            metric_std = [m.std for m in metric_values]
            # print(metric_values)
            try:
                plt.bar(
                    index + i * bar_width,  # Adjust position to avoid overlap
                    metric_mean,
                    bar_width,
                    label=exp.model_name,
                    alpha=0.5,
                )
            except Exception as e:
                print(f'{exp.model_name} caused an error in plot_metric_per_set_size: {e}')
                print(metric_mean)
            
            try:
                plt.errorbar(
                    index + i * bar_width,  # Center error bars on bars
                    metric_mean,
                    yerr=metric_std,  # Include standard deviation as error bars
                    fmt="none",  # No marker for error bars
                    color="black",  # Error bar color
                    capsize=5,  # Cap size for error bars
                    linewidth=1,  # Line width of error bars
                )
            except Exception as e:
                print(f'{exp.model_name} caused an error in plot_metric_per_set_size: {e}')
                print(metric_std)
                
        # Setting the x-axis labels to show set sizes
        plt.xticks(
            index + bar_width / 2 * (len(self.experiments) - 1), index + 1
        )  # +1 because set sizes start from 1

        plt.xlabel("Set Size")
        plt.ylabel(f"{metric.__name__} Value")
        plt.title(
            f"Distribution of {metric.__name__} per set size across experiments \
                   on dataset {self.dataset_name}"
        )
        plt.legend()

        plt.tight_layout()

        if save:
            plt.gcf().set_facecolor("white")
            plt.savefig(
                os.path.join(self.plots_path, f"{metric.__name__}_per_set_size.png"),
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )

        if show:
            plt.show()

    def get_labels_comparison(self, save: bool = False):
        # построить табличку со сравнением метрик относительно меток
        pass

    def get_metrics(self, metric_list: List[Metric]):
        """
        return metrics calculated for whole dataset
        : param metric_list: is expected to be like for e.g. [(roc_auc_score, "probas", "weighted"), (precision_score, "pred_labels", "macro")]
        """

        results = {}
        for exp in self.experiments:
            results[exp.model_name] = exp.get_metrics(metric_list)

        return results

    def to_dataframe(self, results: Dict[str, Dict], save: bool = True) -> pd.DataFrame:
        """saves info about metrics from different models, which are evlauated on same dataset"""
        df = pd.DataFrame.from_dict(results, orient="index").transpose()
        if save:
            df.to_excel(os.path.join(self.data_path, f"{self.dataset_name}.xlsx"))
        return df

    def evaluate_and_save(
        self,
        metric_list_for_set_sizes: List[Metric],
        metric_list_for_labels: List[Metric],
        metric_list: List[Metric],
        figsize: Tuple[int, int] = (18, 10),
        show: bool = False,
    ) -> pd.DataFrame:
        """plots all graphs and evaluates all tables, and saves them"""
        self.plot_info(
            metric_list_for_set_sizes, metric_list_for_labels, figsize=figsize, save=True, show=show
        )
        results = self.get_metrics(metric_list)
        df = self.to_dataframe(results, save=True)
        return df
