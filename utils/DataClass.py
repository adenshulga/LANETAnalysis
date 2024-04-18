import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

import os

from typing import Callable, Tuple


class ProcessedDataset(ABC):
    """
    processed format returned by conf_scores is:
    shape (num_of_samples, num_of_labels)
    each array is expected to be a np.ndarray with confindence scores of the model
    trivial labels must not be excluded

    i may need to handle different models for different datasets
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_name = None
        self.dataset_name = None

    @abstractmethod
    def conf_scores_test(self) -> list:
        pass

    @abstractmethod
    def conf_scores_valid(self) -> list:
        pass

    @abstractmethod
    def probas_test(self) -> list:
        pass

    @abstractmethod
    def probas_valid(self) -> list:
        pass

    @abstractmethod
    def gt_test(self):
        pass

    @abstractmethod
    def gt_valid(self):
        pass


def get_masked(mask, *args):
    """in args expect arrays to get masked"""
    tmp = []
    for arg in args:
        tmp.append(arg[:, mask])
    return tuple(tmp)


def get_thresholded(
    probas: np.ndarray, thresholds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (probas >= thresholds).astype(np.int64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class DataClass:
    def __init__(self, processed: ProcessedDataset) -> None:
        self.conf_scores_test = processed.conf_scores_test()
        self.conf_scores_valid = processed.conf_scores_valid()

        self.gt_valid = processed.gt_valid()
        self.gt_test = processed.gt_test()

        self.dataset_name = processed.dataset_name
        self.model_name = processed.model_name

        self.probas_test = processed.probas_test()
        self.probas_valid = processed.probas_valid()

    def get_thresholds(
        self,
        metric_to_optimize: Callable[[np.ndarray, np.ndarray], float],
        type_of_optim: str = "max",
        is_thresholds_independent: bool = False,
        **kwargs,
    ) -> list[np.ndarray]:
        """
        Adjusted to handle list of probas arrays for multiple launches.
        """
        possible_thr = np.linspace(0, 1, num=100)

        func = np.argmax if type_of_optim == "max" else np.argmin

        thresholds_list = []  # List to store thresholds for each launch

        for launch_index, probas_valid_launch in enumerate(self.probas_valid):  # Iterate over each launch
            if is_thresholds_independent:
                thresholds = np.array(
                    [
                        possible_thr[
                            func(
                                np.array(
                                    [
                                        metric_to_optimize(
                                            self.gt_valid[launch_index][:, j],
                                            probas_valid_launch[:, j] >= thr,
                                            **kwargs,
                                        )
                                        for thr in possible_thr
                                    ]
                                )
                            )
                        ]
                        for j in range(self.gt_valid[launch_index].shape[1])
                    ]
                )
            else:
                hl = []
                for tau in possible_thr:
                    hl.append(
                        metric_to_optimize(
                            y_true=self.gt_valid[launch_index],
                            y_pred=probas_valid_launch >= tau,
                            **kwargs,
                        )
                    )
                min_index = func(np.array(hl))
                opt_tau = possible_thr[min_index]
                thresholds = np.full(self.gt_valid[launch_index].shape[1], opt_tau)

            thresholds_list.append(thresholds)

        return thresholds_list

    def get_non_trivial_targets(self) -> np.ndarray:
        return np.where((self.gt_test[0].sum(axis=0) != 0))[0]

    def show_trivial_info(self) -> None:
        print(f"Num of labels: {self.gt_test[0].shape[1]}")
        print(
            f"Num of trivial labels: {self.gt_test[0].shape[1] - self.get_non_trivial_targets().shape[0]}"
        )

    def get_masked_and_thresholded(
        self, thresholds_list: list[np.ndarray], without_trivial: bool = True
    ) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Adjusted to handle lists of thresholds for multiple launches and correctly utilize the given implementations of get_masked and get_thresholded.
        """
        # results = []  # To store tuples of (gt, scores, probas, labels) for each launch
        gt_list = []
        scores_list = []
        probas_list = []
        labels_list = []

        for launch_index, thresholds in enumerate(thresholds_list):
            # Assuming each element in self.probas_test is a single ndarray for each launch
            probas = self.probas_test[launch_index]
            # Apply thresholding
            labels = get_thresholded(probas, thresholds)

            if without_trivial:
                mask = self.get_non_trivial_targets()
                # Apply masking. Note: scores, probas, and gt need to be masked; labels are already thresholded based on probas
                gt, scores, probas, labels = get_masked(
                    mask,
                    self.gt_test[launch_index],
                    self.conf_scores_test[launch_index],
                    probas,
                    labels,
                )
            else:
                scores = self.conf_scores_test[launch_index]
                gt = self.gt_test

            gt_list.append(gt)
            scores_list.append(scores)
            probas_list.append(probas)
            labels_list.append(labels)
            # results.append((gt, scores, probas, labels))

        return (gt_list, scores_list, probas_list, labels_list)


class ProcessFile(ProcessedDataset):
    # path_to_preds = "/app/All_models/model_pred_and_gt/SFCNTSP"

    def load_from_path(self, dataset_name: str, type: str) -> list:
        """
        Modified to return a list of numpy arrays for each run.
        """
        runs_path = f"{self.path_to_preds}/{dataset_name}"
        data_list = []

        # List all run directories
        runs = [
            d
            for d in os.listdir(runs_path)
            if os.path.isdir(os.path.join(runs_path, d)) and "run_" in d
        ]
        runs.sort(key=lambda x: int(x.split("_")[1]))  # Ensure ordered by run number

        for run in runs:
            path = f"{runs_path}/{run}/{type}/data.csv"
            data = np.genfromtxt(path, delimiter=",")
            data_list.append(data)

        return data_list

    def __init__(self, dataset_name: str, path_to_preds : str, model_name: str) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.path_to_preds = path_to_preds

        self.pred_conf_scores_test = self.load_from_path(dataset_name, "pred_test")
        self.gt_test_data = self.load_from_path(dataset_name, "gt_test")
        self.pred_conf_scores_valid = self.load_from_path(dataset_name, "pred_valid")
        self.gt_valid_data = self.load_from_path(dataset_name, "gt_valid")

    def conf_scores_test(self) -> list:
        "Returns processed to common format np.ndarray"
        return self.pred_conf_scores_test

    def conf_scores_valid(self) -> list:
        return self.pred_conf_scores_valid

    def gt_test(self):
        return self.gt_test_data

    def gt_valid(self):
        return self.gt_valid_data

    def probas_test(self):
        return [sigmoid(data) for data in self.pred_conf_scores_test]

    def probas_valid(self):
        return [sigmoid(data) for data in self.pred_conf_scores_valid]


class ProcessGP(ProcessedDataset):


    def load_from_path(self, dataset_name: str, type: str) -> str:
        "type is gt or pred"
        path = f"{self.path_to_preds}/{dataset_name}/{type}/data.csv"
        return np.genfromtxt(path, delimiter=",")

    def __init__(self, dataset_name: str, path_to_preds: str, model_name: str = "GP_top_freq") -> None:
        self.dataset_name = dataset_name
        self.model_name = model_name

        self.path_to_preds = path_to_preds

        self.pred_conf_scores_test = self.load_from_path(dataset_name, "pred_test")
        self.gt_test_data = self.load_from_path(dataset_name, "gt_test")
        self.pred_conf_scores_valid = self.load_from_path(dataset_name, "pred_valid")
        self.gt_valid_data = self.load_from_path(dataset_name, "gt_valid")

    def conf_scores_test(self) -> np.ndarray:
        "Returns processed to common format np.ndarray"
        return [self.pred_conf_scores_test]

    def conf_scores_valid(self) -> np.ndarray:
        return [self.pred_conf_scores_valid]

    def gt_test(self):
        return [self.gt_test_data]

    def gt_valid(self):
        return [self.gt_valid_data]

    def probas_test(self):
        return [sigmoid(self.pred_conf_scores_test)]

    def probas_valid(self):
        return [sigmoid(self.pred_conf_scores_valid)]
    
# class ProcessLANET(ProcessedDataset):
#     """
#     Example realization of ProcessedDataset subclass.
#     The idea, that this child class incapsulates dataset format converting and loading it from path which is the user specifies
#     """

#     # path_to_preds = f"/app/TCMBN_and_LANET/model_pred_and_gt/LANET"
#     path_to_preds = "/app/All_models/model_pred_and_gt/LANET"


#     def load_from_path(self, dataset_name: str, type: str) -> list:
#         """
#         Modified to return a list of numpy arrays for each run.
#         """
#         runs_path = f"{ProcessLANET.path_to_preds}/{dataset_name}"
#         data_list = []

#         # List all run directories
#         runs = [
#             d
#             for d in os.listdir(runs_path)
#             if os.path.isdir(os.path.join(runs_path, d)) and "run_" in d
#         ]
#         runs.sort(key=lambda x: int(x.split("_")[1]))  # Ensure ordered by run number

#         for run in runs:
#             path = f"{runs_path}/{run}/{type}/data.csv"
#             data = np.genfromtxt(path, delimiter=",")
#             data_list.append(data)

#         return data_list

#     def __init__(self, dataset_name: str, model_name: str = "LANET") -> None:
#         super().__init__()
#         self.dataset_name = dataset_name
#         self.model_name = model_name

#         self.pred_conf_scores_test = self.load_from_path(dataset_name, "pred_test")
#         self.gt_test_data = self.load_from_path(dataset_name, "gt_test")
#         self.pred_conf_scores_valid = self.load_from_path(dataset_name, "pred_valid")
#         self.gt_valid_data = self.load_from_path(dataset_name, "gt_valid")

#     def conf_scores_test(self) -> list:
#         "Returns processed to common format np.ndarray"
#         return self.pred_conf_scores_test

#     def conf_scores_valid(self) -> list:
#         return self.pred_conf_scores_valid

#     def gt_test(self):
#         return self.gt_test_data

#     def gt_valid(self):
#         return self.gt_valid_data

#     def probas_test(self):
#         return [sigmoid(data) for data in self.pred_conf_scores_test]

#     def probas_valid(self):
#         return [sigmoid(data) for data in self.pred_conf_scores_valid]
    
# class ProcessTCMBN(ProcessedDataset):
#     """
#     Example realization of ProcessedDataset subclass.
#     The idea, that this child class incapsulates dataset format converting and loading it from path which is the user specifies
#     """

#     # path_to_preds = f"/app/TCMBN_and_LANET/model_pred_and_gt/TCMBN"
#     path_to_preds = "/app/All_models/model_pred_and_gt/TCMBN"


#     def load_from_path(self, dataset_name: str, type: str) -> list:
#         """
#         Modified to return a list of numpy arrays for each run.
#         """
#         runs_path = f"{ProcessTCMBN.path_to_preds}/{dataset_name}"
#         data_list = []

#         # List all run directories
#         runs = [
#             d
#             for d in os.listdir(runs_path)
#             if os.path.isdir(os.path.join(runs_path, d)) and "run_" in d
#         ]
#         runs.sort(key=lambda x: int(x.split("_")[1]))  # Ensure ordered by run number

#         for run in runs:
#             path = f"{runs_path}/{run}/{type}/data.csv"
#             data = np.genfromtxt(path, delimiter=",")
#             data_list.append(data)

#         return data_list

#     def __init__(self, dataset_name: str, model_name: str = "TCMBN") -> None:
#         super().__init__()
#         self.dataset_name = dataset_name
#         self.model_name = model_name

#         self.pred_conf_scores_test = self.load_from_path(dataset_name, "pred_test")
#         self.gt_test_data = self.load_from_path(dataset_name, "gt_test")
#         self.pred_conf_scores_valid = self.load_from_path(dataset_name, "pred_valid")
#         self.gt_valid_data = self.load_from_path(dataset_name, "gt_valid")

#     def conf_scores_test(self) -> list:
#         "Returns processed to common format np.ndarray"
#         return self.pred_conf_scores_test

#     def conf_scores_valid(self) -> list:
#         return self.pred_conf_scores_valid

#     def gt_test(self):
#         return self.gt_test_data

#     def gt_valid(self):
#         return self.gt_valid_data

#     def probas_test(self):
#         return [sigmoid(data) for data in self.pred_conf_scores_test]

#     def probas_valid(self):
#         return [sigmoid(data) for data in self.pred_conf_scores_valid]


# class ProcessDNNTSP(ProcessedDataset):
#     """
#     Example realization of ProcessedDataset subclass.
#     The idea, that this child class incapsulates dataset format converting and loading it from path which is the user specifies
#     """

#     # path_to_preds = f"/app/DNNTSP/model_pred_and_gt"
#     path_to_preds = "/app/All_models/model_pred_and_gt/DNNTSP"


#     def load_from_path(self, dataset_name: str, type: str) -> list:
#         """
#         Modified to return a list of numpy arrays for each run.
#         """
#         runs_path = f"{ProcessDNNTSP.path_to_preds}/{dataset_name}"
#         data_list = []

#         # List all run directories
#         runs = [
#             d
#             for d in os.listdir(runs_path)
#             if os.path.isdir(os.path.join(runs_path, d)) and "run_" in d
#         ]
#         runs.sort(key=lambda x: int(x.split("_")[1]))  # Ensure ordered by run number

#         for run in runs:
#             path = f"{runs_path}/{run}/{type}/data.csv"
#             data = np.genfromtxt(path, delimiter=",")
#             data_list.append(data)

#         return data_list

#     def __init__(self, dataset_name: str, model_name: str = "DNNTSP") -> None:
#         super().__init__()
#         self.dataset_name = dataset_name
#         self.model_name = model_name

#         self.pred_conf_scores_test = self.load_from_path(dataset_name, "pred_test")
#         self.gt_test_data = self.load_from_path(dataset_name, "gt_test")
#         self.pred_conf_scores_valid = self.load_from_path(dataset_name, "pred_valid")
#         self.gt_valid_data = self.load_from_path(dataset_name, "gt_valid")

#     def conf_scores_test(self) -> list:
#         "Returns processed to common format np.ndarray"
#         return self.pred_conf_scores_test

#     def conf_scores_valid(self) -> list:
#         return self.pred_conf_scores_valid

#     def gt_test(self):
#         return self.gt_test_data

#     def gt_valid(self):
#         return self.gt_valid_data

#     def probas_test(self):
#         return [sigmoid(data) for data in self.pred_conf_scores_test]

#     def probas_valid(self):
#         return [sigmoid(data) for data in self.pred_conf_scores_valid]
    
# # class ProcessTCMBNFile(ProcessedDataset):
# #     """
# #     Example realization of ProcessedDataset subclass.
# #     The idea, that this child class incapsulates dataset format converting and loading it from path which is the user specifies
# #     """

# #     path_to_preds = f"/app/TCMBN/model_pred_and_gt"

# #     def load_from_path(self, dataset_name: str, type: str) -> list:
# #         """
# #         Modified to return a list of numpy arrays for each run.
# #         """
# #         runs_path = f"{ProcessTCMBNFile.path_to_preds}/{dataset_name}"
# #         data_list = []

# #         # List all run directories
# #         runs = [
# #             d
# #             for d in os.listdir(runs_path)
# #             if os.path.isdir(os.path.join(runs_path, d)) and "run_" in d
# #         ]
# #         runs.sort(key=lambda x: int(x.split("_")[1]))  # Ensure ordered by run number

# #         for run in runs:
# #             path = f"{runs_path}/{run}/{type}/data.csv"
# #             data = np.genfromtxt(path, delimiter=",")
# #             data_list.append(data)

# #         return data_list

# #     def __init__(self, dataset_name: str, model_name: str = "DNNTSP") -> None:
# #         super().__init__()
# #         self.dataset_name = dataset_name
# #         self.model_name = model_name

# #         self.pred_conf_scores_test = self.load_from_path(dataset_name, "pred_test")
# #         self.gt_test_data = self.load_from_path(dataset_name, "gt_test")
# #         self.pred_conf_scores_valid = self.load_from_path(dataset_name, "pred_valid")
# #         self.gt_valid_data = self.load_from_path(dataset_name, "gt_valid")

# #     def conf_scores_test(self) -> list:
# #         "Returns processed to common format np.ndarray"
# #         return self.pred_conf_scores_test

# #     def conf_scores_valid(self) -> list:
# #         return self.pred_conf_scores_valid

# #     def gt_test(self):
# #         return self.gt_test_data

# #     def gt_valid(self):
# #         return self.gt_valid_data

# #     def probas_test(self):
# #         return [sigmoid(data) for data in self.pred_conf_scores_test]

# #     def probas_valid(self):
# #         return [sigmoid(data) for data in self.pred_conf_scores_valid]
