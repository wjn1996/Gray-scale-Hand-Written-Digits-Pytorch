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
from Classifier import *

data_path = './data/'
epoch = 10
batch_size = 30 
learn_rate = 0.01
hidden_size = 196
show_every = 10
eval_every = 1000
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(root=data_path, train=True,transform=torchvision.transforms.ToTensor(),              
  download=DOWNLOAD_MNIST)
test_data = torchvision.datasets.MNIST(root=data_path, train=False)
train = MNIST_loader(root='./data/MNIST/processed/training.pt', transform= None)
test = MNIST_loader(root='./data/MNIST/processed/test.pt', transform= None)

train_x, train_y, test_x, test_y = [], [], [], []

for i in train:
    train_x.append(i['img'].numpy())
    train_y.append(i['label'])
for i in test:
    test_x.append(i['img'].numpy())
    test_y.append(i['label'])

# classifier = KNN(train_x, train_y, test_x, test_y)
# classifier = DT(train_x, train_y, test_x, test_y)
classifier = SVM(train_x, train_y, test_x, test_y)
print(classifier.name)
print("training...")
classifier.train()
print("testing...")
acc = classifier.test()
print(classifier.name, ' test acc=', acc)