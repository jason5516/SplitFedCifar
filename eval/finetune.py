import os
from functools import wraps
from collections import defaultdict
from tqdm import tqdm

import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
import copy
import random
import time
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torchvision.models import resnet
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, datasets
from argparse import ArgumentParser
from torchvision import transforms as tt
from PIL import Image
from utils import AverageMeter
import argparse
import pickle
# from torch.utils.data import SequentialSampler


OneLayer = "1_layer"
TwoLayer = "2_layer"
predictor_network=TwoLayer
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        images, labels = self.dataset[self.idxs[item]]
        return images, labels

# With Maxpooling
class BasicBlock(nn.Module):
    # feature expansion
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet
    Note two main differences from official pytorch version:
    1. conv1 kernel size: pytorch version uses kernel_size=7
    2. average pooling: pytorch version uses AdaptiveAvgPool
    """

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.feature_dim = 512 * block.expansion

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # after conv1 do max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64 , num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128 , num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256 , num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512 , num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d((4, 4))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

# augmentation utils
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def byol_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096, class_num=10, num_layer=TwoLayer):
        super().__init__()
        self.in_features = dim
        if num_layer == OneLayer:
            self.net = nn.Sequential(
                nn.Linear(dim, projection_size),
            )
        elif num_layer == TwoLayer:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, projection_size),
            )
        else:
            raise NotImplementedError(f"Not defined MLP: {num_layer}")

    def forward(self, x):
        return self.net(x)
        
class Classifer(nn.Module):
    def __init__(self, dim, hidden_size=1024, class_num=10):
        super().__init__()
        self.in_features = dim
        
        self.fc1 = nn.Linear(dim, hidden_size)
        # self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.softmax = nn.Softmax(dim=1)
        self.fc3 = nn.Linear(hidden_size, class_num)

    def forward(self, x):
        
        x = self.fc1(x)
        # x = self.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        x = self.fc3(x)
        
        return x

class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly 
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                # torchvision.transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
        
def distribute_noniid_traindata_label(labels, num_users, noniid_ratio = 0.55, num_class = 10):
    
    allnum = len(labels)
    finedict = {i: [] for i in range(num_users)}
    traindict = {i: [] for i in range(num_users)}
    classdict = {i: [] for i in range(num_class)}
    targets = np.array(labels)
    targetsid = set(range(allnum))
    if noniid_ratio < 1.0:
        noniid_ratio_other = float(0.45/9)
        num_imgs_per_client = int(allnum / num_users)
        for y in range(num_class):  
            idx = np.where(targets == y)[0]
            classdict[y].extend(idx)
            for i in range(num_users):
                if (i%10) == y:
                    temp = np.random.choice(idx, int(num_imgs_per_client*noniid_ratio), replace=False)
                else:
                    temp = np.random.choice(idx, int(num_imgs_per_client*noniid_ratio_other), replace=False)
                fidx = np.random.choice(temp, int(len(temp)*0.2), replace=False)
                tidx = np.setdiff1d(temp, fidx)
                finedict[i].extend(fidx)
                traindict[i].extend(tidx)
                idx = np.setdiff1d(idx, temp)    
    elif noniid_ratio == 1.0 :
        noniid_ratio_other = 1
        num_imgs_per_client = int(allnum / num_users/ num_class)
        for y in range(num_class):
            idx = np.where(targets == y)[0]
            classdict[y].extend(idx)
            for i in range(num_users):
                temp = np.random.choice(idx, int(num_imgs_per_client), replace=False)
                fidx = np.random.choice(temp, int(len(temp)*0.2), replace=False)
                tidx = np.setdiff1d(temp, fidx)
                finedict[i].extend(fidx)
                traindict[i].extend(tidx)
                idx = np.setdiff1d(idx, temp)    
        
    for i in range(num_users):
            finedict[i] = np.array(finedict[i], dtype=object).flatten()
            traindict[i] = np.array(traindict[i], dtype=object).flatten()

    return traindict, finedict, classdict

def noniid_testdata_label(labels, num_users, noniid_ratio = 0.2, num_class = 10):
    num_class_per_client = int(noniid_ratio * num_class)
    
    num_shards, num_imgs = num_class_per_client * num_users, int(len(labels)/num_users/num_class_per_client)

    idx_shard = np.array([i for i in range(num_shards)])
    idx_shard = idx_shard % num_class
    dict_users_labeled = {i: [] for i in range(num_users)}
    dict_classes_labeled = {i: [] for i in range(num_class)}
    labels_np = np.array(labels)
    for i in range(num_users):
        rand_set = idx_shard[i*num_class_per_client:(i+1)*num_class_per_client]
        for rand in rand_set:
            idx = np.where(labels_np == rand)[0]
            idx = np.random.choice(idx, num_imgs, replace=False)
            dict_users_labeled[i].extend(idx)
        dict_users_labeled[i] = set(np.array(dict_users_labeled[i]))
    
    for i in range(num_class):                                           
        idx = np.where(labels_np == i)[0]
        dict_classes_labeled[i].extend(idx)
        dict_classes_labeled[i] = set(np.array(dict_classes_labeled[i]))

    return dict_users_labeled, dict_classes_labeled

def inference(loader, model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h = model(x)

        h = h.squeeze()
        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 5 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

def get_features(model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, model, device)
    test_X, test_y = inference(test_loader, model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, train_y_dict, classes_dict, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    # Fine tuning by all CIFAR10 training data
    
    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    testing_loader_list = []
    testing_classloader_list = []
    # Fine tuning by split CIFAR10 training data
    if train_y_dict :
        train_loader = []
        for i in range(len(train_y_dict)):
            training_subset = DatasetSplit(train, train_y_dict[i])
            train_loader_temp = torch.utils.data.DataLoader(
                training_subset, batch_size=10, shuffle=False
            )
            train_loader.append(train_loader_temp)
    else:
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=False
        )
        
    # IID test data
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    # non-IID test data
    # for i in range(len(test_y_dict)):
    #     testing_subset = DatasetSplit(test, test_y_dict[i])
    #     test_loader = torch.utils.data.DataLoader(
    #         testing_subset, batch_size=20, shuffle=False
    #     )
    #     testing_loader_list.append(test_loader)
    for i in range(len(classes_dict)):
        testing_subset = DatasetSplit(test, classes_dict[i])
        class_loader = torch.utils.data.DataLoader(
            testing_subset, batch_size=batch_size, shuffle=False
        )
        testing_classloader_list.append(class_loader)
        
    return train_loader, test_loader, testing_classloader_list