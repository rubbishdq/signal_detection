from mod_config import *
from utils import *
from model import ResNet1d
from dataset import SigDetDataset, split_dataset
from torch.utils.data import DataLoader
import torch
import torchvision
import pandas as pd
import shutil
import os
import time
import random

DatasetDir = '../RFML_dataset_new'
ModelDir = '../Models'
ModelFile = None
TestSize = 0.25
SNR = -10
LR = 2e-4
MaxEpoch = 20
BatchSize = 8
BatchNorm = 0
Dropout = 1
DropoutRate = 0.2
PrintPeriod = 100
SaveModel = True
DeleteOldModel = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

fix_all_seeds(2022)

if DeleteOldModel:
    if os.path.exists(ModelDir):
        shutil.rmtree(ModelDir)
    os.mkdir(ModelDir)
else:
    if not os.path.exists(ModelDir):
        os.mkdir(ModelDir)

file_dict = {}
for mod in Mod.keys():
    file_dir = os.path.join(DatasetDir, Mod[mod], f'{mod}_{SNR}dB')
    file_list = os.listdir(file_dir)
    for k in range(len(file_list)):
        file_list[k] = os.path.join(file_dir, file_list[k])
    file_dict[mod] = file_list

# split datasets
train_file_dict, test_file_dict = split_dataset(file_dict, TestSize)
train_ds = SigDetDataset(train_file_dict)
test_ds = SigDetDataset(test_file_dict)
train_dl = DataLoader(dataset=train_ds, batch_size=BatchSize, shuffle=True, drop_last=True)
test_dl = DataLoader(dataset=test_ds, batch_size=BatchSize, shuffle=True, drop_last=True)

print(f'Size of training dataset: {len(train_dl) * BatchSize}')
print(f'Size of test dataset: {len(test_dl) * BatchSize}')

conv_net = ResNet1d(n_class=2, batchnorm=BatchNorm, dropout=Dropout, dropout_rate=DropoutRate)
criterion = torch.nn.CrossEntropyLoss()
conv_optimizer = torch.optim.Adam(conv_net.parameters(), LR)
if ModelFile is not None:
    conv_net.load_state_dict(torch.load(os.path.join(ModelDir, ModelFile), map_location=device))
conv_net = conv_net.to(device)

# print(train_file_dict['Noise'])
# print(test_file_dict['Noise'])

train_losses = []
train_acces = []
test_losses = []
test_acces = []

# Train loop
time_start = time.time()
for e in range(MaxEpoch):
    train_loss_temp = 0
    train_acc_temp = 0
    conv_net.train()
    for i, (data, label) in enumerate(train_dl):
        out = conv_net(data.to(device))
        loss = criterion(out.cpu(), label)
        _, pred = out.max(1)
        # print(train_ds.mod_list)
        # print(out.cpu())
        # print(f'{pred.cpu()}, {label}')
        # print(f'{train_ds.mod_list[int(pred.cpu().numpy())]}, {train_ds.mod_list[int(label.numpy())]}')
        # print(loss.cpu())
        train_loss_temp += loss.item()

        conv_optimizer.zero_grad()
        loss.backward()
        conv_optimizer.step()

        train_acc_temp += np.sum((pred.cpu().numpy() == label.numpy()).astype(np.int32))
        if (i+1) % PrintPeriod == 0:
            print(f'batch id: {e+1} - {i+1}',)
            print(f'train_loss = {train_loss_temp / PrintPeriod}')
            print(f'train_acc = {train_acc_temp / PrintPeriod / BatchSize}')
            # print(out.cpu())
            # print(f'{pred.cpu()}')
            # print(label)
            # print(loss.cpu())
            train_loss_temp = 0
            train_acc_temp = 0
    # save model
    if SaveModel:
        torch.save(conv_net.state_dict(), os.path.join(ModelDir, f'model-e{e+1}.bin'))
    # test once for each epoch
    train_loss = 0
    train_acc = 0
    test_loss = 0
    test_acc = 0
    conv_net.eval()
    for i, (data, label) in enumerate(train_dl):
        out = conv_net(data.to(device))
        loss = criterion(out.cpu(), label)
        train_loss += loss.item()

        _, pred = out.max(1)
        # if (i+1) % PrintPeriod == 0:
        #     print(f'e: {e}  i: {i}')
        #     print(f'out: {out.detach().cpu().numpy()}')
        #     print(f'pred: {pred.cpu().numpy()}')
        #     print(f'label: {label.numpy()}')
        train_acc += np.sum((pred.cpu().numpy() == label.numpy()).astype(np.int32))
    train_losses.append(train_loss / len(train_dl))
    train_acces.append(train_acc / len(train_dl) / BatchSize)
    for i, (data, label) in enumerate(test_dl):
        out = conv_net(data.to(device))
        loss = criterion(out.cpu(), label)
        test_loss += loss.item()

        _, pred = out.max(1)
        test_acc += np.sum((pred.cpu().numpy() == label.numpy()).astype(np.int32))
    test_losses.append(test_loss / len(test_dl))
    test_acces.append(test_acc / len(test_dl) / BatchSize)
    time_now = time.time()
    print(f'Epoch {e+1}')
    print(f'train_loss = {train_loss / len(train_dl)}')
    print(f'train_acc = {train_acc / len(train_dl) / BatchSize}')
    print(f'test_loss = {test_loss / len(test_dl)}')
    print(f'test_acc = {test_acc / len(test_dl) / BatchSize}')
    print(f'time elapsed: {time_now-time_start} s')

df = pd.DataFrame({'Train loss': train_losses, 'Train acc': train_acces,
                   'Test loss': test_losses, 'Test acc': test_acces}, index=range(1, MaxEpoch+1))
print(df)
