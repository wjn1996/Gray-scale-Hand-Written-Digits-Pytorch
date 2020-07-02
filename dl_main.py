import torch
import torchvision 
from torchvision.transforms import transforms
import torch.utils.data as Data 
import scipy.misc
import os
import matplotlib.pyplot as plt   
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from data_loader  import MNIST_loader, showImg
from Network import *
import numpy as np

data_path = './data/'
save_path = './model_save/'
epoch = 20
batch_size = 30 
learn_rate = 0.01
hidden_size = 196
show_every = 10
eval_every = 1000
DOWNLOAD_MNIST = True

def saveLog(train_loss, train_acc, test_loss, test_acc):
    np.save(save_path + 'train_loss.npy', train_loss)
    np.save(save_path + 'train_acc.npy', train_acc)
    np.save(save_path + 'test_loss.npy', test_loss)
    np.save(save_path + 'test_acc.npy', test_acc)

train_data = torchvision.datasets.MNIST(root=data_path, train=True,transform=torchvision.transforms.ToTensor(),              
  download=DOWNLOAD_MNIST)
test_data = torchvision.datasets.MNIST(root=data_path, train=False)
train = MNIST_loader(root='./data/MNIST/processed/training.pt', transform= None)
test = MNIST_loader(root='./data/MNIST/processed/test.pt', transform= None)
# showImg(train)
print('train num:', len(train))
print('test num:', len(test))
width = train[0]['img'].shape[1]
height = train[0]['img'].shape[2]



# net = SingleNN(batch_size=batch_size, width=width, height=height, hidden_size=hidden_size, label_tot=10)
net = CNN(batch_size=batch_size, width=width, height=height, hidden_size=hidden_size, label_tot=10)

optimizer = optim.SGD(net.parameters(), lr=learn_rate)
loss_function = nn.CrossEntropyLoss()

max_acc = 0
print(net.name)
print("training...")

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for i in range(epoch):
	# 使用DataLoader，batch,shuffle等
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

    for ei, batch in enumerate(train_loader):
        batch_x = batch['img']
        batch_y = batch['label']
        net.batch_size = len(batch_y)
        net.zero_grad()
        y_pre = net(batch_x)
        loss = loss_function(y_pre, Variable(batch_y))
        loss.backward()
        optimizer.step()
        pre = torch.argmax(y_pre, axis=1)
        acc_num = (pre == batch_y).sum().float()
        if (ei + 1)% show_every == 0:
            print('[ training epoch:', i+1, ' | batch' , ei+1 ,' | train loss = ', loss, ' | train accruacy = ', acc_num/batch_size, ' ]')

        train_loss.append(loss.detach().numpy().tolist())
        train_acc.append((acc_num/batch_size).numpy().tolist())

        if (ei + 1)% eval_every == 0:
            print("testing...")
            with torch.no_grad():
                acc_sum = 0
                loss_sum = 0
                test_sum = len(test)
                for ei, batch in enumerate(test_loader):
                    batch_x_ = batch['img']
                    batch_y_ = batch['label']
                    net.batch_size = len(batch_y_)
                    y_pre = net(batch_x_)
                    loss = loss_function(y_pre, Variable(batch_y_))
                    pre = torch.argmax(y_pre, axis=1)
                    acc_sum += (pre == batch_y_).sum().float()
                    loss_sum += loss.detach().numpy().tolist()
                test_acc_ = acc_sum/test_sum
                test_loss_ = loss_sum/test_sum
                print('[ testing epoch:', i+1, ' | test loss = ', test_loss_, ' | test accruacy = ', test_acc_, ' ]')
                if max_acc < test_acc_:
                    max_acc = test_acc_
                    torch.save(net.state_dict(), save_path + 'parameter.pkl')

                test_loss.append(test_loss_)
                test_acc.append(test_acc_)
            saveLog(train_loss, train_acc, test_loss, test_acc)

