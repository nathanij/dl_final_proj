import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import torchvision
from glob import glob
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transform
from torch.utils.data import DataLoader,Dataset

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet0(nn.Module):
    def __init__(self):
        super(ResNet0, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.res1 = ResNetBlock(64, 64, 3, 1, 1)
        self.res2 = ResNetBlock(64, 64, 3, 1, 1)
        self.res3 = ResNetBlock(64, 64, 3, 1, 1)
        self.res4 = ResNetBlock(64, 64, 3, 1, 1)
        self.res5 = ResNetBlock(64, 64, 3, 1, 1)
        self.res6 = ResNetBlock(64, 64, 3, 1, 1)
        self.res7 = ResNetBlock(64, 64, 3, 1, 1)
        self.res8 = ResNetBlock(64, 64, 3, 1, 1)
        self.res9 = ResNetBlock(64, 64, 3, 1, 1)
        self.res10 = ResNetBlock(64, 64, 3, 1, 1)
        self.res11 = ResNetBlock(64, 64, 3, 1, 1)
        self.res12 = ResNetBlock(64, 64, 3, 1, 1)
        self.res13 = ResNetBlock(64, 64, 3, 1, 1)
        self.res14 = ResNetBlock(64, 64, 3, 1, 1)
        self.res15 = ResNetBlock(64, 64, 3, 1, 1)
        # turn back int 3x256x256
        self.conv2 = nn.Conv2d(64, 3, 3, 1, 1)



    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)
        out = self.res7(out)
        out = self.res8(out)
        out = self.res9(out)
        out = self.res10(out)
        out = self.res11(out)
        out = self.res12(out)
        out = self.res13(out)
        out = self.res14(out)
        out = self.res15(out)
        out = self.conv2(out)
        return out