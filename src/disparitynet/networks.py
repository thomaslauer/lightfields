import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FullNet(nn.Module):
    def __init__(self, device):
        super(FullNet, self).__init__()
        self.disparity = DisparityNet()
        self.color = ColorNet()
        self.device = device

    def forward(self, x):        
        # bx(200 + 3*4 + 4)x60x60

        # Run disparity
        x = self.disparity(x)
        # TODO: Warp images
        x = self.warp_images(x, 1, 1, 1)
        
        # Compute color from warped images
        x = self.color(x)
        return x

    def warp_images(self, x, img, p_i, q):
        """
        params:
            x: the input disparity for the batch    (N, H, W, 1)
            img: the light field RGB view image     (N, C, H, W)
            p_i: the input image location           (N, H, W, 2)
            q: the novel image location             (N, H, W, 2)
        """

        print(x.shape)

        us = torch.tensor(np.linspace(0, 1, num=x.shape[1]))
        vs = torch.tensor(np.linspace(0, 1, num=x.shape[2]))

        grid_u, grid_v = torch.meshgrid((us, vs))
        grid_u = grid_u.to(self.device)
        grid_v = grid_v.to(self.device)

        grid = torch.stack((grid_u, grid_v))
        print(grid.shape)

        return



class DisparityNet(nn.Module):

    def __init__(self):
        super(DisparityNet, self).__init__()
        self.conv1 = nn.Conv2d(22,  100, kernel_size=(7,7))
        self.conv2 = nn.Conv2d(100, 100, kernel_size=(5,5))
        self.conv3 = nn.Conv2d(100, 50,  kernel_size=(3,3))
        self.conv4 = nn.Conv2d(50,  1,   kernel_size=(1,1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x



class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.conv1 = nn.Conv2d(15,  100, kernel_size=(7,7))
        self.conv2 = nn.Conv2d(100, 100, kernel_size=(5,5))
        self.conv3 = nn.Conv2d(100, 50,  kernel_size=(3,3))
        self.conv4 = nn.Conv2d(50,  3,   kernel_size=(1,1))

    def forward(self, x):
        return x