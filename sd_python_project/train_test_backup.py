from mod_config import *
from utils import *
from dataset import SigDetDataset, split_dataset
from scipy.io import loadmat, savemat
import torch
import torchvision
import numpy as np
import shutil
import os
import random

dataset_dir = '../RFML_dataset_new'
test_size = 0.25
SNR = 20


file_dict = {}
for mod in Mod.keys():
    file_dir = os.path.join(dataset_dir, Mod[mod], f'{mod}_{SNR}dB')
    file_list = []
    file_list = os.listdir(file_dir)
    for k in range(len(file_list)):
        file_list[k] = os.path.join(file_dir, file_list[k])
    file_dict[mod] = file_list

# split datasets
train_file_dict, test_file_dict = split_dataset(file_dict, test_size)
train_ds = SigDetDataset(train_file_dict)
test_ds = SigDetDataset(test_file_dict)
print(f'Size of training dataset: {len(train_ds)}')
print(f'Size of test dataset: {len(test_ds)}')

