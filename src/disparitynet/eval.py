import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt

import datasets
import networks
import utils
import params


def main():
    # parser = argparse.ArgumentParser(description="Evaluates the naive CNN")
    # parser.add_argument('epoch', type=int)
    # parser.add_argument('lightfield', type=str)

    lightFieldPaths = [
        # "../datasets/reflective_17_eslf.png",
        # "../datasets/reflective_18_eslf.png",
        # "../datasets/flowers_plants/raw/flowers_plants_9_eslf.png"
        "../../../datasets/microcropped_images/flowers_plants_25_eslf.png"
    ]

    epochNum = 36

    full_dataset = datasets.LytroDataset(lightFieldPaths, training=False, cropped=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    net = networks.FullNet(device)
    net.load_state_dict(torch.load(utils.get_checkpoint_path(epochNum)))
    net = net.to(device)
    net.eval()

    for i, (depth, color) in enumerate(tqdm(full_dataset)):
        x = i // 8 + 1
        y = i % 8 + 1
        depth = torch.unsqueeze(torch.Tensor(depth), 0).to(device)
        color = torch.unsqueeze(torch.Tensor(color), 0).to(device)

        images = color[:, :-2, :, :]
        novelLocation = color[:, -2:, :, :]

        disp = net.disparity(depth).cpu()
        dispImg = utils.torch2np_color(disp[0].detach().numpy())

        output = net(depth, images, novelLocation).cpu()
        img = utils.torch2np_color(output[0].detach().numpy())
        img = utils.adjust_tone(img)
        utils.mkdirp(f"output/epoch_{epochNum}/color")
        utils.mkdirp(f"output/epoch_{epochNum}/disp")
        imageio.imwrite(f"output/epoch_{epochNum}/color/nn_0{y}_0{x}.png", img)
        imageio.imwrite(f"output/epoch_{epochNum}/disp/nn_0{y}_0{x}.png", dispImg)

    return


if __name__ == "__main__":
    main()
