import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import imageio
from tqdm import tqdm

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
        "../datasets/flowers_plants/raw/flowers_plants_9_eslf.png"
    ]

    full_dataset = datasets.LytroDataset(lightFieldPaths, training=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    net = networks.NaiveNet().half()
    net.load_state_dict(torch.load(utils.get_checkpoint_path(0)))
    net = net.to(device)
    net.half()
    net.eval()

    for i, (data, _) in enumerate(tqdm(full_dataset)):
        x = i // 8 + 1
        y = i % 8 + 1
        data = torch.unsqueeze(torch.Tensor(data), 0)
        data = data.to(device).half()
        output = net(data).cpu()
        print(type(output))
        img = utils.torch2np_color(output[0].detach().numpy() ** (1/2.2))
        imageio.imwrite(f"output/nn_0{y}_0{x}.png", img)


    return

if __name__ == "__main__":
    main()