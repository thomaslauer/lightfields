import torch
import numpy as np
from tqdm import tqdm

import datasets
import networks
import utils
import params


def load_network(save_epoch):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    net = networks.FullNet(device)
    net.load_state_dict(torch.load(utils.get_checkpoint_path(save_epoch), map_location=torch.device(device)))
    # net = torch.jit.trace(net, (torch.rand(1, 200, 376, 541), torch.rand(
    #     1, 4 * 5, 376 - 12, 541 - 12), torch.rand(1, 2, 376 - 12, 541 - 12)))
    # torch.jit.save(net, "backups/main_model.pt")
    net = net.to(device)
    net.eval()
    return net, device


def eval_net(net, depth, color):
    with torch.no_grad():
        depth, color = net.single_input(depth, color)

        disp, warps, output = net.all_steps(depth, color)

        output = output[0].cpu()
        disp = disp[0, 0].cpu()
        warped = utils.stack_warps(warps[0, :12].cpu().numpy())
        dispImg = disp.detach().numpy()
        img = utils.torch2np_color(output.detach().numpy())
        rawImg = img
        img = utils.adjust_tone(img)
        warped = utils.adjust_tone(warped)

        return img, dispImg, rawImg, warped


def save_outputs(epoch, name, imgs):
    img, dispImg, rawImg, warp = imgs
    utils.mkdirp(f"output/epoch_{epoch}/color")
    utils.mkdirp(f"output/epoch_{epoch}/disp")
    utils.mkdirp(f"output/epoch_{epoch}/warp")
    utils.mkdirp(f"output/epoch_{epoch}/raw")

    utils.save_image(f"output/epoch_{epoch}/color/{name}", img)
    utils.save_disparity(f"output/epoch_{epoch}/disp/{name}", dispImg)
    utils.save_image(f"output/epoch_{epoch}/raw/{name}", rawImg)
    utils.save_image(f"output/epoch_{epoch}/warp/{name}", warp)

def novel_view():
    data = datasets.LytroDataset(["../datasets/microcropped_images/Rock.png"], training=False, cropped=True)
    net, _ = load_network(17)
    for i in tqdm(range(20, 40)):
        disp, color = data.completely_novel_view(0, i / 7, 0.5)
        images = eval_net(net, disp, color)

        name = f"strange_{i:02}.png"
        save_outputs("test", name, images)
    print("finished generating novel views")
    exit()


def main():
    # parser = argparse.ArgumentParser(description="Evaluates the naive CNN")
    # parser.add_argument('epoch', type=int)
    # parser.add_argument('lightfield', type=str)

    lightFieldPaths = [
        # "../datasets/reflective_17_eslf.png",
        # "../datasets/reflective_18_eslf.png",
        # "../datasets/flowers_plants/raw/flowers_plants_25_eslf.png"
        "../datasets/microcropped_images/Rock.png"
        # f"{params.drive_path}/datasets/microcropped_images/flowers_plants_9_eslf.png"
    ]

    epochNum = 17
    net, device = load_network(epochNum)

    full_dataset = datasets.LytroDataset(lightFieldPaths, training=False, cropped=True)

    for i, (depth, color, _target) in enumerate(tqdm(full_dataset)):
        x = i // 8 + 1
        y = i % 8 + 1

        outputs = eval_net(net, depth, color)

        name = f"nn_0{y}_0{x}.png"
        save_outputs(epochNum, name, outputs)


if __name__ == "__main__":
    # novel_view()
    main()
