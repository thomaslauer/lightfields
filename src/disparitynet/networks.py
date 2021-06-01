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

    def forward(self, disparityFeatures: torch.Tensor, colorFeatures: torch.Tensor):
        # bx(200 + 3*4 + 4)x60x60

        # disparityFeatures: (batch x 200 x H x W)
        # images: (batch x RGBUV x W-6 x H-6)
        # novelLocation: (batch x W-6 x H-6 x UV)
        # result: RGB x W-12 x H-12

        # Run disparity
        return self.all_steps(disparityFeatures, colorFeatures)[2]

    @torch.jit.export
    def all_steps(self, disparityFeatures, colorFeatures):
        images = colorFeatures[:, :-2, :, :]
        novelLocation = colorFeatures[:, -2:, :, :]

        disparity: torch.Tensor = self.disparity(disparityFeatures)
        # TODO: Warp images
        warps = self.warp_images(disparity, images, novelLocation)

        # Compute color from warped images
        finalImg: torch.Tensor = self.color(warps)
        return disparity, warps, finalImg

    def single_input(self, disp, color):
        depth = torch.unsqueeze(torch.as_tensor(disp, device=self.device), 0)
        color = torch.unsqueeze(torch.as_tensor(color, device=self.device), 0)
        return depth, color

    def warp_images(self, disp, images, novelLocation):
        """
        params:
            disp: the input disparity for the batch         (N, 1, H, W)
            img: the light field RGBUV view image           (N, RGBUV, H, W)
            p_i: the input image location                   (N, H, W, 2)
            q: the novel image location                     (N, H, W, 2)
        """

        # make (N, H, W, 1)
        disparity = torch.moveaxis(disp, 1, -1)

        batches, rows, cols, _1 = disparity.shape

        # duplicate disparity on last axis so it matches p_i and q formats
        # make (N, H, W, 2)
        # Note: due to a really big dumb dumb, we forgot to scale the u/v coords of the meshgrid
        # to be in the range [0, D) so the disparity was actually trained to be in the wrong range
        # and would only work correctly for 60x60 source -> 48x48, so all the disparity values were off by a scale of 47.
        # to fix it, we add the scale to the disparity here, and then that cancels out the grid scale normalization.
        # NOTE: followup fixed and retrained
        dupedDisparity = disparity.repeat(1, 1, 1, 2)

        # (grid + disp * const) / [rows - 1, cols - 1]

        # U is down, V is right
        us = torch.linspace(0, rows-1, rows, dtype=torch.float32)
        vs = torch.linspace(0, cols-1, cols, dtype=torch.float32)

        grid_u, grid_v = torch.meshgrid(us, vs)
        grid_u = grid_u.to(self.device)
        grid_v = grid_v.to(self.device)

        # build "s" grid of pixel locations to sample
        # [N, H, W, UV]
        grid = torch.stack((grid_u, grid_v), dim=-1)
        grid = torch.unsqueeze(grid, dim=0)
        grid = grid.repeat(batches, 1, 1, 1)

        # novelLocation = torch.reshape(
        #     novelLocation, (novelLocation.shape[0], novelLocation.shape[2], novelLocation.shape[3], novelLocation.shape[1]))
        reshapedNovelLocation = torch.moveaxis(novelLocation, 1, -1)
        # reshapedNovelLocation = reshapedNovelLocation[:, :rows, :cols, :]

        warpedImages = []

        for i in range(4):
            off = 5 * i

            # Get the color data
            currentImg = images[:, off:off+3, :, :]
            # Get the u, v coords (N, UV, H, W)
            p_i = images[:, off+3:off+5, :, :]

            # [N, H, W, UV]
            p_i = torch.moveaxis(p_i, 1, -1)

            # ???
            # p_i = p_i[:, :rows, :cols, :]

            projectedLocations = grid + (p_i - reshapedNovelLocation) * dupedDisparity
            projectedLocations[..., 0] /= rows - 1
            projectedLocations[..., 1] /= cols - 1
            projectedLocations = (projectedLocations - 0.5) * 2
            # print(projectedLocations.requires_grad)

            # [N, H, W, VU] =~ [N, H, W, XY]
            # Locations are expected in XY, where X => W, Y => H, so opposite of our UV coords :'(
            # projectedLocations = projectedLocations[..., -1:0:-1]
            projectedLocations = torch.flip(projectedLocations, [-1])

            warpedImg = F.grid_sample(currentImg, projectedLocations, mode='bicubic', align_corners=False)
            warpedImages.append(warpedImg)

        warpedImages.append(disp)
        warpedImages.append(novelLocation)
        stacked = torch.cat(warpedImages, dim=1)

        return stacked


class DisparityNet(nn.Module):

    def __init__(self):
        super(DisparityNet, self).__init__()
        self.conv1 = nn.Conv2d(200,  100, kernel_size=(7, 7))
        self.conv2 = nn.Conv2d(100, 100, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(100, 50,  kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(50,  1,   kernel_size=(1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.conv1 = nn.Conv2d(15,  100, kernel_size=(7, 7))
        self.conv2 = nn.Conv2d(100, 100, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(100, 50,  kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(50,  3,   kernel_size=(1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x
