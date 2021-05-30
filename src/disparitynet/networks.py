import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class FullNet(nn.Module):
    def __init__(self, device):
        super(FullNet, self).__init__()
        self.disparity = DisparityNet()
        self.color = ColorNet()
        self.device = device

    def forward(self, disparityFeatures, images, novelLocation):
        # bx(200 + 3*4 + 4)x60x60

        # disparityFeatures: (batch x 200 x H x W)
        # images: (batch x RGBUV x W-6 x H-6)
        # novelLocation: (batch x W-6 x H-6 x UV)
        # result: RGB x W-12 x H-12

        # Run disparity
        x = self.disparity(disparityFeatures)
        # TODO: Warp images
        x = self.warp_images(x, images, novelLocation)
        
        # Compute color from warped images
        x = self.color(x)
        return x

    def warp_images(self, x, images, novelLocation):
        """
        params:
            x: the input disparity for the batch            (N, 1, H, W)
            img: the light field RGBUV view image           (N, RGBUV, H, W)
            p_i: the input image location                   (N, H, W, 2)
            q: the novel image location                     (N, H, W, 2)
        """

        # rearrange disparity to match the format (N, H, W, 1) format
        dispShape = x.shape
        # disparity = torch.reshape(x, (dispShape[0], dispShape[2], dispShape[3], dispShape[1]))
        disparity = torch.moveaxis(x, 1, -1)

        us = torch.tensor(np.linspace(0, 1, num=disparity.shape[1]))
        vs = torch.tensor(np.linspace(0, 1, num=disparity.shape[2]))

        grid_u, grid_v = torch.meshgrid(us, vs)
        grid_u = grid_u.to(self.device)
        grid_v = grid_v.to(self.device)

        # build "s" grid of pixel locations to sample
        grid = torch.stack((grid_v, grid_u), dim=-1)
        grid = torch.unsqueeze(grid, dim=0)
        grid = grid.repeat(disparity.shape[0], 1, 1, 1)

        # duplicate disparity on last axis so it matches p_i and q formats
        dupedDisparity = disparity.repeat(1, 1, 1, 2)

        # novelLocation = torch.reshape(
        #     novelLocation, (novelLocation.shape[0], novelLocation.shape[2], novelLocation.shape[3], novelLocation.shape[1]))
        reshapedNovelLocation = torch.moveaxis(novelLocation, 1, -1)
        reshapedNovelLocation = reshapedNovelLocation[:, :disparity.shape[1], :disparity.shape[2], :]

        warpedImages = []

        for i in range(4):
            currentImg = images[:, 5*i:5*i+3, :, :]
            p_i = images[:, 5*i+3:5*i+5, :, :]

            # p_i = torch.reshape(p_i, (p_i.shape[0], p_i.shape[2], p_i.shape[3], p_i.shape[1]))
            p_i = torch.moveaxis(p_i, 1, -1)

            p_i = p_i[:, :disparity.shape[1], :disparity.shape[2], :]

            projectedLocations = grid + (p_i - reshapedNovelLocation) * dupedDisparity
            projectedLocations = (projectedLocations - 0.5) * 2
            # print(projectedLocations.shape)

            warpedImg = F.grid_sample(currentImg, projectedLocations.float(), mode='bicubic')
            warpedImages.append(warpedImg)

            # plt.subplot(2, 2, 1)
            # plt.imshow(dupedDisparity.detach().cpu().numpy()[0][:,:,0])
            # plt.subplot(2, 2, 2)
            # plt.imshow(projectedLocations.detach().cpu().numpy()[0][:,:,0])
            # plt.subplot(2, 2, 3)
            # plt.imshow(warpedImg.detach().cpu().numpy()[0][0,:,:])
            # plt.show()

            # plt.imshow(np.moveaxis(warpedImg.detach().cpu().numpy()[0], 0, -1))
            # plt.imshow(p_i.detach().cpu().numpy()[0])
            # plt.show()

        # fig, axs = plt.subplots(2, 2)
        # for ax, img in zip(axs.flatten(), warpedImages):
        #     ax.imshow(np.moveaxis(img.detach().cpu().numpy()[0], 0, -1))
        # plt.show()

        warpedImages.append(x)
        warpedImages.append(novelLocation)
        stacked = torch.cat(warpedImages, dim=1)

        return stacked



class DisparityNet(nn.Module):

    def __init__(self):
        super(DisparityNet, self).__init__()
        self.conv1 = nn.Conv2d(200,  100, kernel_size=(7,7))
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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x
