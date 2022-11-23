from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import torch
import numpy as np
import pandas as pd
import random


# for_dl: used for initializing dataloader
class SigDetDataset(Dataset):
    def __init__(self, file_dict, shuffle=True, for_dl=True):
        self.data_list = []  # list of dicts
        self.for_dl = for_dl
        self.mod_list = list(file_dict.keys())
        for mod in file_dict.keys():
            for file in file_dict[mod]:
                data_ele = {}
                data_ele['data_path'] = file
                data_ele['mod_type'] = mod
                self.data_list.append(data_ele)
        if shuffle:
            random.shuffle(self.data_list)

    def __getitem__(self, idx):
        mat_data = loadmat(self.data_list[idx]['data_path'])
        if self.for_dl:
            signal = torch.from_numpy(mat_data['IQ']).to(torch.float32)
            mod_type = self.data_list[idx]['mod_type']
            # label = torch.tensor(self.mod_list.index(mod_type), dtype=torch.long)
            label = torch.tensor(int(mod_type=='Noise'), dtype=torch.long)
            return signal, label
        else:
            signal = mat_data['IQ']
            sig_range = mat_data['sig_range'] if ('sig_range' in mat_data.keys()) else None
            mod_type = self.data_list[idx]['mod_type']
            data_info = {'sig_range': sig_range,
                         'mod_type': mod_type}
            return signal, data_info

    def __len__(self):
        return len(self.data_list)


# return splitted datasets (in the form of file_dict)
def split_dataset(file_dict, test_size):
    file_list = []  # list of dicts
    mod_list = []
    for mod in file_dict.keys():
        file_list += file_dict[mod]
        mod_list += ([mod] * len(file_dict[mod]))
    file_train, file_test, mod_train, mod_test = train_test_split(file_list, mod_list, test_size=test_size,
                                                                  shuffle=True, stratify=mod_list, random_state=2023)
    return zip_dataset(file_train, mod_train), zip_dataset(file_test, mod_test)


def zip_dataset(files, mods):
    file_dict = {}
    for k in range(len(files)):
        if mods[k] in file_dict.keys():
            file_dict[mods[k]].append(files[k])
        else:
            file_dict[mods[k]] = [files[k]]
    return file_dict

