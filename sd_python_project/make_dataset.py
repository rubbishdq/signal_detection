from mod_config import *
from utils import *
from scipy.io import loadmat, savemat
import torch
import torchvision
import numpy as np
import shutil
import os
import random

fix_all_seeds(2022)

DatasetDir = '../RFML_dataset'
TargetDir = '../RFML_dataset_new'
N_DataUsed = 256   # determine how much data is used
N_Noise = N_DataUsed
SigLen = 1024
SNR = -10
P_LenRange = (1., 1.0)
P_NoiseRange = (1., 1.)
UseOldDataWithSNR = True    # if true, use original old data; if false, manually add noise to 30dB data

if os.path.exists(TargetDir):
    shutil.rmtree(TargetDir)
os.mkdir(TargetDir)

file_dict = {}
SNR_data_file = SNR if UseOldDataWithSNR else 30
for mod in Mod.keys():
    if mod == 'Noise':
        continue
    file_dir = os.path.join(DatasetDir, Mod[mod], f'{mod}_{SNR_data_file}dB')
    file_list = []
    for k in range(N_DataUsed):
        file_list.append(f'{mod}_{SNR_data_file}dB_{k+1}.mat')
    # file_list = os.listdir(file_dir)
    # for k in range(len(file_list)):
    #     file_list[k] = os.path.join(file_dir, file_list[k])
    file_dict[mod] = file_list

s_dataset = 0
for mod in Mod.keys():
    if mod == 'Noise':
        continue
    s_dataset += len(file_dict[mod])

avg_p_noise = {}
for mod in Mod.keys():
    type_dir = os.path.join(TargetDir, Mod[mod])
    if not os.path.exists(type_dir):
        os.mkdir(type_dir)
    file_dir = os.path.join(type_dir, f'{mod}_{SNR_data_file}dB')
    os.mkdir(file_dir)
    if mod == 'Noise':
        continue
    else:
        avg_p_noise_temp = 0.
        for file in file_dict[mod]:
            old_data_file = os.path.join(DatasetDir, Mod[mod], f'{mod}_{SNR_data_file}dB', file)
            new_data_file = os.path.join(file_dir, file)
            signal_data = loadmat(old_data_file)['IQ']
            new_sig, sig_range, p_noise = add_noise(signal_data, SNR, P_LenRange, P_NoiseRange,
                                                    use_old_data=UseOldDataWithSNR)
            data_dict = {'IQ': new_sig, 'sig_range': sig_range}
            savemat(new_data_file, data_dict)
            avg_p_noise_temp += p_noise
        avg_p_noise[mod] = (avg_p_noise_temp / len(file_dict[mod]))
        print(f'{mod} data has been processed.')
print(avg_p_noise)

type_dir = os.path.join(TargetDir, '0.Noise')
if not os.path.exists(type_dir):
    os.mkdir(type_dir)
file_dir = os.path.join(type_dir, f'Noise_{SNR_data_file}dB')
file_id = 0
for mod in Mod.keys():
    if mod == 'Noise':
        continue
    for k in range(N_Noise):
        new_data_file = os.path.join(file_dir, f'Noise_{SNR_data_file}dB_{file_id+1}.mat')
        new_len = int(SigLen * np.random.uniform(P_LenRange[0], P_LenRange[1]))
        p_noise = avg_p_noise[mod] * np.random.uniform(P_NoiseRange[0], P_NoiseRange[1])
        noise = np.random.normal(0., np.sqrt(p_noise / 2.), (2, new_len))
        sig_range = np.array([-1, -1])
        data_dict = {'IQ': noise, 'sig_range': sig_range}
        savemat(new_data_file, data_dict)
        file_id += 1
print(f'Noise data has been generated.')

print(f'\nNew dataset has been saved to {os.path.realpath(TargetDir)}.')
