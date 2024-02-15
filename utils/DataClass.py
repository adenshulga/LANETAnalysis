import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from typing import Callable, Tuple

class ProcessedDataset(ABC):
    '''
    processed format returned by conf_scores is:
    shape (num_of_samples, num_of_labels)
    each array is expected to be a np.ndarray with confindence scores of the model
    trivial labels must not be excluded

    i may need to handle different models for different datasets
    '''
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
    ''' in args expect arrays to get masked'''
    tmp = []
    for arg in args:
        tmp.append(arg[:, mask])
    return tuple(tmp)

def get_thresholded(probas : np.ndarray,
                    thresholds: np.ndarray
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

    def get_thresholds(self, 
                   metric_to_optimize: Callable[[np.ndarray, np.ndarray], float], 
                   type_of_optim: str='max', is_thresholds_independent: bool=False, **kwargs) -> np.ndarray:
        '''
        type_of_optim is max or min
        threshold should be retrieved on validation
        '''
        possible_thr = np.linspace(0, 1, num=100)

        if type_of_optim == 'max':
            func = np.argmax
        elif type_of_optim == 'min':
            func = np.argmin
        
        if is_thresholds_independent:
            self.thresholds = np.array([possible_thr[func(np.array([metric_to_optimize(self.gt_valid[:, j], self.probas_valid[:, j] >= thr, **kwargs) 
                                                            for thr in possible_thr]))] for j in range(self.gt_valid.shape[1])])
        else:
            hl = []
            for tau in possible_thr:
                hl.append(metric_to_optimize(y_true=self.gt_valid, y_pred=np.array(self.probas_valid)>tau, **kwargs))
               # hl.append(hamming_loss(y_true=true_label, y_pred=np.array(pred_label)>tau))
            min_index = func(np.array(hl))
            opt_tau = possible_thr[min_index]
            self.thresholds = np.full(self.gt_valid.shape[1], opt_tau)

        return self.thresholds
    
    def get_non_trivial_targets(self) -> np.ndarray:
        return np.where((self.gt_test.sum(axis=0) != 0))[0]
    
    def show_trivial_info(self) -> None:
        print(f'Num of labels: {self.gt_test.shape[1]}')
        print(f'Num of trivial labels: {self.gt_test.shape[1] - self.get_non_trivial_targets().shape[0]}')

    
    def get_masked_and_thresholded(self, 
                                   thresholds: np.ndarray, 
                                   without_trivial: bool=True
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        scores = self.conf_scores_test
        probas = self.probas_test
        gt = self.gt_test
        labels = get_thresholded(probas, thresholds)

        if without_trivial:
            mask = self.get_non_trivial_targets()
            scores, labels, probas, gt = get_masked(mask, scores, labels, probas, gt)
            

        return gt, scores, probas, labels
 
class ProcessSFCNTSPFile(ProcessedDataset):
    
    path_to_preds = f'/home/andrey/Prgrm/mllab/FlashBackupRIP/SFCNTSP-master/model_pred_and_gt'

    def load_from_path(self, dataset_name: str, type: str) -> str:
        'type is gt or pred'
        path = f'{ProcessSFCNTSPFile.path_to_preds}/{dataset_name}/{type}/data.csv'
        return np.genfromtxt(path, delimiter=',')

    def __init__(self, dataset_name: str, model_name: str='SFCNTSP') -> None:

        self.dataset_name = dataset_name
        self.model_name = model_name

        self.pred_conf_scores_test = self.load_from_path(dataset_name, 'pred_test')
        self.gt_test_data = self.load_from_path(dataset_name, 'gt_test')
        self.pred_conf_scores_valid = self.load_from_path(dataset_name, 'pred_validate')
        self.gt_valid_data = self.load_from_path(dataset_name, 'gt_validate')

    def conf_scores_test(self) -> np.ndarray:
        'Returns processed to common format np.ndarray'
        return self.pred_conf_scores_test
    
    def conf_scores_valid(self) -> np.ndarray:
        return self.pred_conf_scores_valid
    
    def gt_test(self):
        return self.gt_test_data
    
    def gt_valid(self):
        return self.gt_valid_data
    
    def probas_test(self):
        return self.pred_conf_scores_test

    def probas_valid(self):
        return self.pred_conf_scores_valid
    
class ProcessGP(ProcessedDataset):

    path_to_preds = f'/home/andrey/Prgrm/mllab/FlashBackupRIP/MyGP_topfreq/model_pred_and_gt'

    def load_from_path(self, dataset_name: str, type: str) -> str:
        'type is gt or pred'
        path = f'{ProcessSFCNTSPFile.path_to_preds}/{dataset_name}/{type}/data.csv'
        return np.genfromtxt(path, delimiter=',')

    def __init__(self, dataset_name: str, model_name: str='GP_top_freq') -> None:

        self.dataset_name = dataset_name
        self.model_name = model_name

        self.pred_conf_scores_test = self.load_from_path(dataset_name, 'pred_test')
        self.gt_test_data = self.load_from_path(dataset_name, 'gt_test')
        self.pred_conf_scores_valid = self.load_from_path(dataset_name, 'pred_validate')
        self.gt_valid_data = self.load_from_path(dataset_name, 'gt_validate')

    def conf_scores_test(self) -> np.ndarray:
        'Returns processed to common format np.ndarray'
        return self.pred_conf_scores_test
    
    def conf_scores_valid(self) -> np.ndarray:
        return self.pred_conf_scores_valid
    
    def gt_test(self):
        return self.gt_test_data
    
    def gt_valid(self):
        return self.gt_valid_data
    
    def probas_test(self):
        return sigmoid(self.pred_conf_scores_test)

    def probas_valid(self):
        return sigmoid(self.pred_conf_scores_valid)

class ProcessDNNTSPFile(ProcessedDataset):
    '''
    Example realization of ProcessedDataset subclass.
    The idea, that this child class incapsulates dataset format converting and loading it from path which is the user specifies
    '''

    path_to_preds = f'/home/andrey/Prgrm/mllab/FlashBackupRIP/DNNTSP/model_pred_and_gt'


    def load_from_path(self, dataset_name: str, type: str) -> str:
        'type is gt or pred'
        path = f'{ProcessDNNTSPFile.path_to_preds}/{dataset_name}/{type}/data.csv'
        return np.genfromtxt(path, delimiter=',')

    def __init__(self, dataset_name: str, model_name: str='DNNTSP') -> None:

        self.dataset_name = dataset_name
        self.model_name = model_name

        self.pred_conf_scores_test = self.load_from_path(dataset_name, 'pred_test')
        self.gt_test_data = self.load_from_path(dataset_name, 'gt_test')
        self.pred_conf_scores_valid = self.load_from_path(dataset_name, 'pred_valid')
        self.gt_valid_data = self.load_from_path(dataset_name, 'gt_valid')

    def conf_scores_test(self) -> np.ndarray:
        'Returns processed to common format np.ndarray'
        return self.pred_conf_scores_test
    
    def conf_scores_valid(self) -> np.ndarray:
        return self.pred_conf_scores_valid
    
    def gt_test(self):
        return self.gt_test_data
    
    def gt_valid(self):
        return self.gt_valid_data
    
    def probas_test(self):
        return sigmoid(self.pred_conf_scores_test)

    def probas_valid(self):
        return sigmoid(self.pred_conf_scores_valid)

    

        

    
        


    