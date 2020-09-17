import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from util import *

"""
Our modified resnet18 model without transfer learning
"""
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.features = models.resnet18()
        self.features.layer3 = nn.Sequential()
        self.features.layer4 = nn.Sequential()
        self.features.fc = nn.Linear(128, 10)
        self.features.relu = nn.LeakyReLU(inplace=True)
        self.features = self.features
    def forward(self, x):
        x = self.features(x)
        return x
    
 
"""
Our modified resnet18 model with transfer learning.
"""
class ResNet18_tr(nn.Module):
    def __init__(self):
        super(ResNet18_tr, self).__init__()
        model_ft = models.resnet18(pretrained=True)
        set_parameter_requires_grad(model_ft, True)
        self.features = nn.Sequential(
            *list(model_ft.children())[:6]
        )
        self.conv1 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 2, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.bh1 = nn.BatchNorm2d(num_features = 256)
        self.pool3 = nn.AvgPool2d(kernel_size = (2,2))
        
        self.lin1 = nn.Linear(in_features = 1024, out_features =500)
        self.lin2 = nn.Linear(in_features = 500, out_features = 10)

    def forward(self, x):
        x = self.features(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(self.bh1(x))
        x = self.pool3(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.lin1(x)
        x = self.lin2(x)  
        return x
