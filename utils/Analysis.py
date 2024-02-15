import numpy as np
import pandas as pd
from typing import Any, Tuple, Callable, Dict, List
from utils.DataClass import DataClass

import os

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, hamming_loss, precision_score, recall_score

import matplotlib.pyplot as plt

def get_label_frequencies(y: np.ndarray) -> np.ndarray:
    return y.sum(axis=0)/y.shape[0]

def plot_label_distribution(gt: np.ndarray, 
                            *args: np.ndarray, 
                            figsize: Tuple[int, int] = (18, 10)) -> None:
    '''this function supports plotting of distribution of multiple arrays'''
    # Calculate the number of labels
    num_labels = gt.shape[1]
    print(num_labels)
    # Calculate label occurrences for each dataset
    occurrences = [gt.sum(axis=0) / gt.shape[0]]

    for arg in args:
        occurrences.append(arg.sum(axis=0) / arg.shape[0])
    
    # Determine the number of groups and bar width
    n_groups = num_labels
    n_labels = len(occurrences)
    bar_width = 1 / (n_labels + 1)
    # Create figure
    plt.figure(figsize=figsize)
    # Create an index for each group
    index = np.arange(n_groups)
    # Plot each dataset
    for i, occ in enumerate(occurrences):
        plt.bar(index + i * bar_width, occ, bar_width, label=f'Dataset {i}', alpha=0.5)

    plt.xlabel('Label Index')
    plt.ylabel('Frequency')
    plt.title('Distribution of Label Frequencies')
    plt.xticks(index + bar_width / 2 * (n_labels - 1), index)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_probas_distribution(probas: np.ndarray, index_of_label: int) -> None:

    plt.hist(probas[:, index_of_label])
    plt.title(f'Distribution of predicted probas for label {index_of_label}')

    plt.show()

def kl_divergence(gt: np.ndarray, pred_labels: np.ndarray) -> float:
    '''
    notation:
    p - corresponds to gt
    q - corresponds to our prediction
    '''
    p = get_label_frequencies(gt)
    q = get_label_frequencies(pred_labels)

    kl_div = np.mean(np.where( (p != 0) & (q != 0), p * np.log( p / q), 0))
    return kl_div

def avg_size_of_pred_set(y: np.ndarray) -> float:
    return np.mean(np.sum(y, axis=1))


class Metric:
    def __init__(self, 
                 metric_func: Callable[[np.ndarray, np.ndarray],  float], 
                 required_arr: str,
                 name: str=None,
                 **kwargs) -> None:
        self.metric_func = metric_func
        self.required_arr = required_arr
        self.kwargs = kwargs 
        
        if name is None:
            # print(metric_func.__name__)
            # print(len(kwargs))
            # print(len(kwargs) == 0)
            
            if len(kwargs) == 0:
                self.__name__ = metric_func.__name__
            else:
                self.__name__ = f'{metric_func.__name__}_{kwargs}'
            
            # print(self.__name__)
        else:
            self.__name__ = name

        
    def __call__(self, gt, pred_labels, probas) -> float:
        if self.required_arr == 'probas':
            values = probas
        elif self.required_arr == 'pred_labels':
            values = pred_labels
        return self.metric_func(gt, values, **self.kwargs)

class ExperimentInfo:

    class LabelInfo:

        def __init__(self, 
                     gt: np.ndarray, 
                     scores: np.ndarray, 
                     probas: np.ndarray, 
                     pred_labels: np.ndarray) -> None:
            '''
            Note: args scores, probas provided for class interface consistency and fo future usage
            This proxy class incapsulates per label analysis and operations.
            '''
            freqs_gt = get_label_frequencies(gt)
            freqs_pred = get_label_frequencies(pred_labels)
            df = pd.DataFrame()
            sorted_indices = np.argsort(freqs_gt)[::-1]
            self.sorted_indices = sorted_indices
            self.num_of_labels = gt.shape[1]

            for i in sorted_indices:
                label_gt = gt[:, i]
                label_pred = pred_labels[:, i]


                new_row = {
                'Label': i,
                'Frequency of the label': freqs_gt[i],
                'Prediction label frequency': freqs_pred[i],
                'Precision': precision_score(label_gt, label_pred),
                'Recall': recall_score(label_gt, label_pred),
                'Accuracy' : accuracy_score(label_gt, label_pred)
                }
                df = df.append(new_row, ignore_index=True)
            
            self.df = df
            self.freqs = freqs_gt[sorted_indices]
            self.gt, self.scores, self.probas, self.pred_labels = gt[:,sorted_indices], scores[:,sorted_indices], probas[:,sorted_indices], pred_labels[:,sorted_indices] 

        def metric_per_label(self, 
                             metric: Metric) -> list:
            ''' 
            required_arr is "probas" or "pred_labels"
            metric_func expects first argument to be gt labels, and second one predicted values
            '''
            
            metric_values = [metric(self.gt[:, i], self.pred_labels[:, i], self.probas[:, i]) for i in range(self.num_of_labels)]

            return metric_values

    class SetInfo:
        '''
        aggregated statistics per sets:
        - metric per set size
        - distribution of set sizes
        - distriubtion of set sizes differences
        '''
        
        def __init__(self, 
                     gt: np.ndarray, 
                     scores: np.ndarray, 
                     probas: np.ndarray, 
                     pred_labels: np.ndarray) -> None:
            
            self.gt, self.scores, self.probas, self.pred_labels = gt, scores, probas, pred_labels

            self.gt_set_sizes = np.sum(gt, axis=1).astype(int)
            self.pred_set_sizes = np.sum(pred_labels, axis=1).astype(int)
            self.max_set_size =  np.max(self.gt_set_sizes).astype(int)
            
            # because of the filtration we don't have set with 0 size in gt
            self.set_sizes = list(range(1, self.max_set_size + 1)) 
            self.set_sizes_distribution = np.bincount(self.gt_set_sizes)

        def metric_per_set_size(self, 
                                metric: Metric) -> list:
            '''
            :param required_arr: is "probas" or "pred_labels"
            
            not all metrics and their averages are supported, 
            because not all labels are present for e.g. in records with set size 1
            '''

            metric_values = []

            for size in self.set_sizes:
                indices = np.where(self.gt_set_sizes == size)
                
                non_constant_label_indices = [i for i in range(self.gt.shape[1])
                                      if len(np.unique(self.gt[indices, i])) > 1]
                
                if len(non_constant_label_indices) == 0:
                    metric_values.append(0)
                    continue 
                
                gt = self.gt[indices][:, non_constant_label_indices]
                labels = self.pred_labels[indices][:, non_constant_label_indices]
                probas = self.probas[indices][:,non_constant_label_indices]

                metric_values.append(
                    metric(gt, labels, probas)
                )


            return metric_values
        
    def get_metrics(self,
                    metric_list: List[Metric]) -> dict:
        '''
        : param metric_list: is expected to be like for e.g. [(roc_auc_score, "probas", "weighted"), (precision_score, "pred_labels", "macro")]
        '''
        result = {}
        for metric in metric_list:
            result[metric.__name__] = metric(self.gt, self.pred_labels, self.probas)
        
        return result
        
        


    def __init__(self, data: DataClass) -> None:
        thresholds = data.get_thresholds(f1_score, 'max', is_thresholds_independent=True, average='weighted')
        self.gt, self.scores, self.probas, self.pred_labels = data.get_masked_and_thresholded(thresholds)

        self.model_name = data.model_name
        self.dataset_name = data.dataset_name

        self.label_info = ExperimentInfo.LabelInfo(self.gt, self.scores, self.probas, self.pred_labels)
        self.set_info = ExperimentInfo.SetInfo(self.gt, self.scores, self.probas, self.pred_labels)


class ModelComparison:
    
    def __init__(self, *args: DataClass, path: str = './') -> None:
        '''
        : param path: path where to create "data" directory with all plots and results. On default, directory is created in one where the script is invoked
        '''

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

        self.plots_path = os.path.join(path, f'data/{self.dataset_name}/plots')
        self.data_path = os.path.join(path, f'data/{self.dataset_name}')

        os.makedirs(self.plots_path, exist_ok=True)

    def check_data(self) -> bool:

        is_ok = True

        # check whether we work on dataset with same name
        dataset_names = [exp.dataset_name for exp in self.experiments ]
        if len(set(dataset_names)) != 1:
            print("ERROR: Check the DataClass objects, \
                  it seems they are from different datasets and therefore we can't compare them")
            print(dataset_names)
            is_ok = False
        else:
            print("Experiments performed on dataset with common name")

        # check that frequencies from gt are same
        freq_arrs = [ exp.label_info.freqs for exp in self.experiments]
        if not all(np.array_equal(freq_arrs[0], freq_arr) for freq_arr in freq_arrs):
            print("ERROR: Seems that gt differs between experiments")
            is_ok = False
        else:
            print("Gt frequencies are equal between experiments")

        max_set_sizes = [exp.set_info.max_set_size for exp in self.experiments]
        if len(set(max_set_sizes)) != 1:
            print("ERROR: Max set sizes are unequal between datasets")
            print(max_set_sizes)
            is_ok = False
        else:
            print("Max set sizes are equal between datasets")

        set_sizes_distributions = [exp.set_info.set_sizes_distribution for exp in self.experiments]
        if not all(np.array_equal(set_sizes_distributions[0], set_sizes_distribution) for set_sizes_distribution in set_sizes_distributions):
            print("ERROR: Set sizes distributions in gt are not equal")
            is_ok = False
        else:
            print("Ground truth set sizes distributions are equal") 

        return is_ok

    def plot_info(self, 
                  metric_list: List[Metric],
                  metric_list_for_labels: List[Metric],
                  figsize: Tuple[int,int]=(18,10),
                  save: bool=False, 
                  show: bool=True) -> None:
        '''plots multiple canvases with different metrics'''
        for metric in metric_list_for_labels:
            self.plot_metric_per_label(metric, figsize=figsize, save=save, show=show)
        
        self.plot_label_distribution(figsize=figsize, save=save, show=show)
        
        for metric in metric_list:
            self.plot_metric_per_set_size(metric, save=save, show=show)
            
        self.plot_set_sizes_distribution(figsize=figsize, save=save, show = show)
        self.plot_set_sizes_differences_distribution(figsize=figsize, save=save, show = show)
        self.plot_per_basket_errors_distribution(figsize=figsize, save=save, show=show)

    def plot_label_distribution(self, 
                                figsize: Tuple[int, int] = (18, 10), 
                                save: bool=False, 
                                show: bool=True) -> None:
        '''this function supports plotting of distribution of multiple arrays'''
        # Calculate the number of labels
        gt = self.experiments[0].gt
        num_labels = gt.shape[1]
        print(num_labels)
        # Calculate label occurrences for each dataset
        occurrences = [gt.sum(axis=0) / gt.shape[0]]
        model_names = ['Ground Truth']
        for exp in self.experiments:
            occurrences.append(exp.pred_labels.sum(axis=0) / exp.pred_labels.shape[0])
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
            plt.bar(index + i * bar_width, occ, bar_width, label=f'{model_names[i]}', alpha=0.5)

        plt.xlabel('Label Index')
        plt.ylabel('Frequency')
        plt.title('Distribution of Label Frequencies')
        plt.xticks(index + bar_width / 2 * (n_labels - 1), index)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.plots_path, f'Label distribution on dataset {self.dataset_name}.png'), dpi=300, bbox_inches='tight', facecolor='white')

        
        if show:
            plt.show()

    def plot_metric_per_label(self, 
                              metric: Metric, 
                              figsize: Tuple[int, int]=(18,10),
                              save: bool=False, 
                              show: bool=True) -> None:
        '''
        plots multiple distributions on one canvas with specified metric
        
        :param metric: A metric function that takes ground truth and predicted labels as input.
        :param required_arr: is "probas" or "pred_labels"
        '''

        plt.figure(figsize=figsize)
        plt.grid(True)

        # Get the number of labels from the first experiment (assuming all have the same number)
        # num_labels = self.experiments[0].label_info.num_of_labels
        bar_width = 1 / (len(self.experiments) + 1)
        index = np.arange(self.num_of_labels)

        # Iterate over each experiment and plot the metric
        for i, exp in enumerate(self.experiments):
            metric_values = exp.label_info.metric_per_label(metric)
            plt.bar(index + i * bar_width, metric_values, bar_width, label=exp.model_name, alpha=0.5)

        # Setting the x-axis labels to show label frequencies
        label_frequencies = self.experiments[0].label_info.freqs
        plt.xticks(index + bar_width / 2 * (len(self.experiments) - 1), label_frequencies)

        plt.xlabel('Label Frequency')
        plt.ylabel(f'{metric.__name__} Value')
        plt.title(f'Distribution of {metric.__name__} per label across experiments on dataset {self.dataset_name}')
        plt.legend()
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.plots_path, f'{metric.__name__}_per_label.png'), dpi=300, bbox_inches='tight', facecolor='white')

        if show:
            plt.show()

    
    def plot_set_sizes_distribution(self, 
                                    figsize: Tuple[int, int] = (18, 10), 
                                    save: bool=False, 
                                    show: bool=True) -> None:
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
        bar_width = 1 / (len(self.experiments) + 2)  # +2 for ground truth and extra spacing

        # Plotting the distribution of set sizes for ground truth
        gt_set_sizes_distribution = [np.sum(self.experiments[0].set_info.gt_set_sizes == i) for i in index]
        plt.bar(index - bar_width, gt_set_sizes_distribution, bar_width, label='Ground Truth', alpha=0.5, color='g')

        # Iterate over each experiment and plot the distribution of set sizes
        for i, exp in enumerate(self.experiments):
            pred_set_sizes_distribution = [np.sum(exp.set_info.pred_set_sizes == j) for j in index]
            plt.bar(index + i * bar_width, pred_set_sizes_distribution, bar_width, label=exp.model_name, alpha=0.5)

        # Setting the x-axis labels to show set sizes
        plt.xticks(index, index)

        plt.xlabel('Set size')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of set sizes in ground truth and across experiments on dataset {self.dataset_name}')
        plt.legend()
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.plots_path, f'set_sizes_distribution.png'), dpi=300, bbox_inches='tight' , facecolor='white')

        if show:
            plt.show()

    def plot_set_sizes_differences_distribution(self, 
                                                figsize: Tuple[int, int]=(18,10), 
                                                save: bool=False, 
                                                show: bool=True) -> None:
        """
        Plots the distribution of set size differences between predictions and ground truth for each dataset in self.experiments.

        :param figsize: Size of the plot.
        """

        plt.figure(figsize=figsize)
        plt.grid(True)

        # Find the maximum possible set size difference
        max_set_size_difference = max([exp.set_info.max_set_size for exp in self.experiments])

        # Create an index array for plotting
        index = np.arange(max_set_size_difference + 1)  # +1 to include the max difference

        # Determine bar width
        bar_width = 1 / (len(self.experiments) + 1)  # +1 for extra spacing

        # Iterate over each experiment and plot the distribution of set size differences
        for i, exp in enumerate(self.experiments):
            # Calculate set size differences
            set_size_diffs = np.abs(np.sum(exp.gt, axis=1) - np.sum(exp.pred_labels, axis=1))
            set_size_diffs_distribution = [np.sum(set_size_diffs == j) for j in index]

            # Plotting
            plt.bar(index + i * bar_width, set_size_diffs_distribution, bar_width, label=exp.model_name, alpha=0.5)

        # Setting the x-axis labels to show set size differences
        plt.xticks(index, index)

        plt.xlabel('Set size difference')
        plt.ylabel('Frequency')
        plt.title('Distribution of set size differences across experiments')
        plt.legend()
        plt.tight_layout()

        if save:
            plt.gcf().set_facecolor('white')
            plt.savefig(os.path.join(self.plots_path, f'set_sizes_difference_distribution.png'), dpi=300, bbox_inches='tight', facecolor='white')

        if show:
            plt.show()

    def plot_per_basket_errors_distribution(self, 
                                            figsize: Tuple[int, int]=(18,10),
                                            save: bool=False,  
                                            show: bool=True) -> None:
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
            errors_per_basket = np.sum(np.abs(exp.gt - exp.pred_labels), axis=1)
            errors_distribution = [np.sum(errors_per_basket == j) for j in index]

            # Plotting
            plt.bar(index + i * bar_width, errors_distribution, bar_width, label=exp.model_name, alpha=0.5)

        # Setting the x-axis labels to show number of errors per basket
        plt.xticks(index, index)

        plt.xlabel('Number of errors per basket')
        plt.ylabel('Frequency')
        plt.title('Distribution of per-basket errors across experiments')
        plt.legend()
        plt.tight_layout()

        if save:
            # plt.gcf().set_facecolor('white')
            plt.savefig(os.path.join(self.plots_path, f'per_basket_errors_distribution.png'), dpi=300, bbox_inches='tight', facecolor='white')

        if show:
            plt.show()


    def plot_metric_per_set_size(self, 
                                 metric: Metric,
                                 figsize: Tuple[int, int]=(18,10), 
                                 save: bool=False, 
                                 show: bool=True) -> None:
        '''
        Plots multiple distributions of a specified metric per set size on one canvas.

        :param metric: A metric function that takes ground truth and predicted labels as input.
        :param required_arr: is "probas" or "pred_labels"
        :param figsize: Size of the plot.
        '''

        plt.figure(figsize=figsize)
        plt.grid(True)

        # Determine the number of set sizes and calculate the bar width
        num_set_sizes = self.max_set_size
        bar_width = 1 / (len(self.experiments) + 1)
        index = np.arange(num_set_sizes)

        # Iterate over each experiment and plot the metric for each set size
        for i, exp in enumerate(self.experiments):
            # try:
            metric_values = exp.set_info.metric_per_set_size(metric)
                # print(metric_values)
            plt.bar(index + i * bar_width, metric_values, bar_width, label=exp.model_name, alpha=0.5)
            # except ValueError:
            #     pass # not all metrics can work with our data, 
            #          # so we just won't plot corresponding graph in case of a problem

        # Setting the x-axis labels to show set sizes
        plt.xticks(index + bar_width / 2 * (len(self.experiments) - 1), index + 1) # +1 because set sizes start from 1

        plt.xlabel('Set Size')
        plt.ylabel(f'{metric.__name__} Value')
        plt.title(f'Distribution of {metric.__name__} per set size across experiments \
                   on dataset {self.dataset_name}')
        plt.legend()
        
        plt.tight_layout()

        if save:
            plt.gcf().set_facecolor('white')
            plt.savefig(os.path.join(self.plots_path, f'{metric.__name__}_per_set_size.png'), dpi=300, bbox_inches='tight', facecolor='white')

        if show:
            plt.show()


    def get_labels_comparison(self, 
                              save: bool=False):
        # построить табличку со сравнением метрик относительно меток
        pass

    def get_metrics(self, 
                    metric_list: List[Metric]):
        '''
        return metrics calculated for whole dataset
        : param metric_list: is expected to be like for e.g. [(roc_auc_score, "probas", "weighted"), (precision_score, "pred_labels", "macro")]
        '''

        results = {}
        for exp in self.experiments:
            results[exp.model_name] = exp.get_metrics(metric_list)

        return results
    
    def to_dataframe(self, 
                     results: Dict[str, Dict],
                     save: bool=True) -> pd.DataFrame:
        '''saves info about metrics from different models, which are evlauated on same dataset'''
        df = pd.DataFrame.from_dict(results, orient='index').transpose()
        if save:
            df.to_excel(os.path.join(self.data_path, f'{self.dataset_name}.xlsx'))
        return df
        

    def evaluate_and_save(self, 
                          metric_list: List[Metric],
                          metric_list_for_labels: List[Metric],
                          figsize: Tuple[int,int]=(18,10),
                          show: bool=False) -> pd.DataFrame:
        '''plots all graphs and evaluates all tables, and saves them'''
        self.plot_info(metric_list, metric_list_for_labels, figsize=figsize, save=True, show=show)
        results = self.get_metrics(metric_list)
        df = self.to_dataframe(results, save=True)
        return df













