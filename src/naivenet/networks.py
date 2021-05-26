import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveNet(nn.Module):

    def __init__(self):
        super(NaiveNet, self).__init__()
        self.conv1 = nn.Conv2d(22,  200, kernel_size=(7,7))
        self.conv2 = nn.Conv2d(200, 100, kernel_size=(5,5))
        self.conv3 = nn.Conv2d(100, 50,  kernel_size=(3,3))
        self.conv4 = nn.Conv2d(50,  3,   kernel_size=(1,1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


