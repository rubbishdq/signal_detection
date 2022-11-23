import torch
import numpy as np
import random
import os

def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def add_noise(signal, SNR, p_len_range=(1., 1.), p_noise_range = (1., 1.),
              mu=0.0, use_old_data=False):
    sig_len = np.size(signal, 1)
    new_len = int(sig_len*np.random.uniform(p_len_range[0], p_len_range[1]))
    sig_start = np.random.randint(0, new_len-sig_len+1)
    sig_end = sig_start + sig_len
    p_sig = np.sum(signal**2) / sig_len
    SNR_nl = np.power(10.0, SNR/10.0)
    p_noise = p_sig / (SNR_nl+1) * np.random.uniform(p_noise_range[0], p_noise_range[1])
    #print(p_sig, p_noise)

    new_sig = np.random.normal(mu, np.sqrt(p_noise/2.), (2, new_len))
    if use_old_data:
        new_sig[:, sig_start:sig_end] = signal
    else:
        new_sig[:, sig_start:sig_end] += signal
    return new_sig, np.array([sig_start, sig_end], np.float32), p_noise
