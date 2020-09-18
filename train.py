import torchvision.datasets as datasets
import torch.utils.data as data
from torchvision import models
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import platform
import time
import copy
import random
from torchvision import models
from util import *
from model import *

#Reading the data from Cifar and perform data augmentation and create dataloaders.
transform_1 = transforms.Compose(
    [transforms.RandomRotation(20),
     transforms.RandomHorizontalFlip(0.5),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset_1 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_1)


train_size = int(0.8 * len(trainset_1))
test_size = len(trainset_1) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(trainset_1, [train_size, test_size])

trainloader_1 = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                          shuffle=True, num_workers=0, worker_init_fn=2)

valloader_1 = torch.utils.data.DataLoader(val_dataset, batch_size=32,
                                          shuffle=True, num_workers=0, worker_init_fn=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=0)

#define parameters for training the ResNet18 model without tranfer learning
model_1 = ResNet18()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_1.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32
save_path = "./ResNet18.pt"

model_resnet18,train_acc, train_loss,val_acc,val_loss = train_model(model_1, criterion, optimizer, exp_lr_scheduler,
                                                              trainloader_1, valloader_1, batch_size, 
                                                              device, save_path, num_epochs=30)

#define parameters for training the ResNet18_tr model with tranfer learning
model_2 = ResNet18_tr()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_2.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32
save_path = "./ResNet18_tl.pt"

model_resnet18_tl,train_acc, train_loss,val_acc,val_loss = train_model(model_2, criterion, optimizer,
                                                                       exp_lr_scheduler,trainloader_1, valloader_1, 
                                                                       batch_size,device, save_path, num_epochs=30)
