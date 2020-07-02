import torch
import torchvision 
from torchvision.transforms import transforms
import torch.utils.data as Data 
import scipy.misc
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image


class MNIST_loader(Dataset):
    def __init__(self, root,  transform=None):
        self.transform = transform
        self.data, self.labels = torch.load(root)
    def __getitem__(self, index):
        # 用来获取一些索引的数据，使dataset[i]返回数据集中第i个样本。
        img, label = self.data[index], int(self.labels[index])
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
        img =transforms.ToTensor()(img)

        sample = {'img': img, 'label': label}
        return sample
    def __len__(self):
        # 实现len(dataset)返回整个数据集的大小
        return len(self.data)

def showImg(dataset):
    for (cnt,i) in enumerate(dataset):
        image = i['img']
        label = i['label']
        ax = plt.subplot(4, 4, cnt+1)
        # ax.axis('off')
        ax.imshow(image.squeeze(0))
        ax.set_title(label)
        plt.pause(0.001)
        if cnt ==15:
            plt.savefig('./MNIST_case.jpg')
            break
