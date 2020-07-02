import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SingleNN(nn.Module):
    # 只含有一层隐层的神经网络
    # epoch=10, accuracy=93.86%
    def __init__(self, batch_size, width, height, hidden_size, label_tot):
        super(SingleNN, self).__init__()
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.hidden_size = hidden_size
        self.label_tot = label_tot
        self.name = 'Single Neural Network'

        self.linear1 = nn.Linear(self.width*self.height, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.label_tot)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # print(x.shape)
        # x: [batch_size, 1, width, height]
        x = x.squeeze(1).view(self.batch_size, -1) # x: [batch_size, width*height]
        out = F.relu(self.dropout(self.linear1(x)))
        out = self.linear2(out)
        return out


class CNN(nn.Module):
	# 两层CNN网络 + 一层神经网络
	# epoch=10, acc=98.59%
    def __init__(self, batch_size, width, height, hidden_size, label_tot):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.hidden_size = hidden_size
        self.label_tot = label_tot
        self.name = 'Convolutional Neural Network (2-layer)'

        self.conv1 = nn.Sequential(	# [batch_size, 32, 28, 28]
            nn.Conv2d(in_channels=1,
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [batch_size, 16, 14, 14]
        )

        self.conv2 = nn.Sequential(	# [batch_size, 64, 14, 14]
            nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=2,
                            stride=1,
                            padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [batch_size, 64, 7, 7]
        )



        self.linear1 = nn.Linear(64*7*7, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.label_tot)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # print(x.shape)
        # x: [batch_size, 1, width, height]
        cnn1_out = self.conv1(x)
        cnn2_out = self.conv2(cnn1_out)
        out = cnn2_out.view(cnn2_out.shape[0], -1)
        # print(cnn2_out.shape)
        out = F.relu(self.dropout(self.linear1(out)))
        out = self.linear2(out)
        return out


