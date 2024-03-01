import json
import os
import pickle

import numpy as np

import random

def unpickle_file(path, type_of_split, data, prefix, encoding):
    tmp_path = os.path.join(path, prefix + type_of_split + '.pkl')
    with open(tmp_path, 'rb') as file:
        data[type_of_split] = pickle.load(file, encoding=encoding)[type_of_split]
    return data

def retrieve_dict(path, prefix, encoding):
    tmp_path = os.path.join(path, prefix + 'dev.pkl')
    print(encoding)
    with open(tmp_path, 'rb') as file:
        data = pickle.load(file, encoding=encoding)
    unpickle_file(path, 'train', data, prefix, encoding=encoding)
    unpickle_file(path, 'test', data, prefix, encoding=encoding)
    return data

def change_busket_enc_format(one_hot_encoded: np.ndarray) -> np.ndarray:
    '''if the busket is empty'''
    return np.nonzero(one_hot_encoded)[0].tolist()

def retrieve_busket_seq(TCMBN_data_dict_record: list) -> list:
    '''
    TCMBN_data_dict_record is expected to be list with info corresponding to one user id
    '''
    busket_seq = []

    for record in TCMBN_data_dict_record:
        one_hot = record['type_event']
        if ~(np.all(one_hot == 0)):
            busket_seq.append(change_busket_enc_format(one_hot))
        busket_seq.append(change_busket_enc_format(one_hot))
    
    return busket_seq

def create_DNNTSP_dict(TCMBN_arr: list) -> dict:
    '''
    TCMBN_merged_arr is expected to be a list with data
    TCMBN_merged_arr[i] returns data regarding i-th id  
    '''

    DNNTSP_dict = {}

    for id, record in enumerate(TCMBN_arr):
        busket_seq = retrieve_busket_seq(record)
        if busket_seq:  # Only add non-empty sequences
            DNNTSP_dict[str(id)] = busket_seq

    return DNNTSP_dict

def save_TCMBN_to_DNNTSP_format(dataset_name: str, path_to_pickled_files: str, ratio=[0.6,0.2,0.2], prefix='', encoding='ASCII') -> None:

    splits = {'train' : 'train', 'test' : 'test', 'dev' : 'validate'}

    TCMBN_unpickled_data = retrieve_dict(path_to_pickled_files, prefix, encoding=encoding)
    DNNTSP_dict = {}
    for key, value in splits.items(): 
        DNNTSP_dict[value] = create_DNNTSP_dict(TCMBN_unpickled_data[key])
    
    new_path = os.path.join('data', dataset_name)
    os.makedirs(new_path, exist_ok=True)

    with open(os.path.join(new_path, dataset_name) + '.json', 'w') as file:
        json.dump(DNNTSP_dict, file)

    return DNNTSP_dict

# Example

# dataset_name = 'Mimic3'
# path_to_pickled_files = '/notebook/tcmbn_data-server/mimic3/split_1'
# new_dict = save_TCMBN_to_DNNTSP_format(dataset_name, path_to_pickled_files, prefix='', encoding='latin-1', seed=42)























