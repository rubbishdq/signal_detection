# Reference: https://blog.csdn.net/weixin_36979214/article/details/108879684
import torch
import torch.nn as nn
from torch.nn import functional as F


class ResNet1dBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, batchnorm=0, dropout=0, dropout_rate=0.2):
        super(ResNet1dBasicBlock, self).__init__()
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout2d(p=dropout_rate, inplace=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout2d(p=dropout_rate, inplace=False)

    def forward(self, x):
        output = self.conv1(x)
        if self.batchnorm > 0:
            output = self.bn1(output)
        if self.dropout > 0:
            output = self.dropout1(output)
        output = F.relu(output)
        output = self.conv2(output)
        if self.batchnorm > 0:
            output = self.bn2(output)
        if self.dropout > 0:
            output = self.dropout2(output)
        return F.relu(x + output)


class ResNet1dDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, batchnorm=0, dropout=0, dropout_rate=0.2):
        super(ResNet1dDownBlock, self).__init__()
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout2d(p=dropout_rate, inplace=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout2d(p=dropout_rate, inplace=False)
        self.extra = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm1d(out_channels),
            nn.Dropout2d(p=dropout_rate, inplace=False)
        )
        if self.batchnorm == 1:
            if self.dropout == 1:
                self.extra = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
                    nn.BatchNorm1d(out_channels),
                    nn.Dropout2d(p=dropout_rate, inplace=False)
                )
            else:
                self.extra = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
                    nn.BatchNorm1d(out_channels)
                )
        else:
            if self.dropout == 1:
                self.extra = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
                    nn.Dropout2d(p=dropout_rate, inplace=False)
                )
            else:
                self.extra = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0)
                )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        if self.batchnorm > 0:
            output = self.bn1(output)
        if self.dropout > 0:
            output = self.dropout1(output)
        output = F.relu(output)

        output = self.conv2(output)
        if self.batchnorm > 0:
            output = self.bn2(output)
        if self.dropout > 0:
            output = self.dropout2(output)
        return F.relu(extra_x + output)


class ResNet1d(nn.Module):  # input: 1024    output: ?
    def __init__(self, n_class=10, batchnorm=0, dropout=0, dropout_rate=0.2):
        super(ResNet1d, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=9, stride=3, padding=4)   # 342
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)  # 114

        self.layer1 = nn.Sequential(ResNet1dBasicBlock(64, 64, 1, batchnorm=batchnorm, dropout=dropout, dropout_rate=dropout_rate),
                                    ResNet1dBasicBlock(64, 64, 1, batchnorm=batchnorm, dropout=dropout, dropout_rate=dropout_rate))

        self.layer2 = nn.Sequential(ResNet1dDownBlock(64, 128, [2, 1], batchnorm=batchnorm, dropout=dropout, dropout_rate=dropout_rate),
                                    ResNet1dBasicBlock(128, 128, 1, batchnorm=batchnorm, dropout=dropout, dropout_rate=dropout_rate))

        self.layer3 = nn.Sequential(ResNet1dDownBlock(128, 256, [2, 1], batchnorm=batchnorm, dropout=dropout, dropout_rate=dropout_rate),
                                    ResNet1dBasicBlock(256, 256, 1, batchnorm=batchnorm, dropout=dropout, dropout_rate=dropout_rate))

        self.layer4 = nn.Sequential(ResNet1dDownBlock(256, 512, [2, 1], batchnorm=batchnorm, dropout=dropout, dropout_rate=dropout_rate),
                                    ResNet1dBasicBlock(512, 512, 1, batchnorm=batchnorm, dropout=dropout, dropout_rate=dropout_rate))

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)

        self.fc = nn.Linear(512, n_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out
