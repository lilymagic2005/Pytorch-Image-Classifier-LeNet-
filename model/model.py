from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models
from config import config
from torch.nn import functional
device = config.device

class LeNet(nn.Module):
    def __init__(self , num_class):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10,num_class)

    def forward(self,x):
        x = functional.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]        #x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    if __name__ == "__main__":
        a = 0
        #print("")
