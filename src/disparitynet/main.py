import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import glob

import datasets
import networks
import params
import utils
import imageio


def train(net, train_loader, optimizer, device, epoch):

    net.train()

    print(f"Training epoch {epoch}")
    for depth, color, target in tqdm(train_loader):
        optimizer.zero_grad()

        depth = depth.to(device)
        color = color.to(device)
        target = target.to(device)

        images = color[:, :-2, :, :]
        novelLocation = color[:, -2:, :, :]

        predicted = net(depth, images, novelLocation)

        loss = F.mse_loss(predicted, target)
        loss.backward()
        optimizer.step()


def saveModel(net, epoch):
    utils.mkdirp("./checkpoints")
    torch.save(net.state_dict(), utils.get_checkpoint_path(epoch))


def validate(net, validation_loader, device, epoch):

    loss_sum = 0

    net.eval()

    with torch.no_grad():
        for depth, color, target in tqdm(validation_loader):

            depth = depth.to(device)
            color = color.to(device)
            target = target.to(device)

            images = color[:, :-2, :, :]
            novelLocation = color[:, -2:, :, :]

            predicted = net(depth, images, novelLocation)
            loss = F.mse_loss(predicted, target)

            loss_sum += torch.sum(loss.cpu())

        # TODO: add other eval functions here

    return loss_sum


def test_image(net, depth, color, device, epoch, out_folder="./eval_test"):
    utils.mkdirp(out_folder)
    net.eval()
    with torch.no_grad():
        depth = torch.unsqueeze(torch.Tensor(depth), 0).to(device)
        color = torch.unsqueeze(torch.Tensor(color), 0).to(device)

        images = color[:, :-2, :, :]
        novelLocation = color[:, -2:, :, :]

        output = net(depth, images, novelLocation).cpu()
        img = utils.torch2np_color(output[0].detach().numpy())
        imageio.imwrite(f"{out_folder}/epoch_{epoch:03}.png", utils.adjust_tone(img))


def main():

    # determine whether to use cuda or cpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # lightFieldPaths = [
    #     "../datasets/reflective_17_eslf.png",
    #     "../datasets/reflective_18_eslf.png",
    # ]

    print("Building training data")
    lightFieldPaths = glob.glob(f'{params.drive_path}/datasets/microcropped_images/*.png')
    full_dataset = datasets.LytroDataset(lightFieldPaths, training=True, cropped=True)
    flower_9 = f'{params.drive_path}/datasets/microcropped_images/flowers_plants_9_eslf.png'

    print("Building test data")
    # Fetch test data
    test_data = datasets.LytroDataset([flower_9], training=False, cropped=True)
    test_depth, test_color = test_data[36]
    # clear memory if possible...
    test_data = None
    print("Splitting train/validate")

    train_size = int(0.9 * len(full_dataset))
    validate_size = len(full_dataset) - train_size

    train_dataset, validate_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, validate_size]
    )

    batch_size = 64
    workers = 3

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=workers)
    validate_loader = DataLoader(validate_dataset, shuffle=True, batch_size=batch_size, num_workers=workers)

    net = networks.FullNet(device)

    if params.start_epoch != 0:
        # load previous epoch checkpoint
        net.load_state_dict(torch.load(utils.get_checkpoint_path(params.start_epoch-1)))

    # move net to cuda BEFORE setting optimizer variables
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=params.sgd_lr, momentum=params.sgd_momentum)
    if params.start_epoch != 0:
        test_image(net, test_depth, test_color, device, params.start_epoch - 1, out_folder="./eval_test")

    for epoch in range(params.start_epoch, params.start_epoch + params.epochs):
        train(net, train_loader, optimizer, device, epoch)
        saveModel(net, epoch)
        loss = validate(net, validate_loader, device, epoch)
        print(f"Validation loss is {loss}")
        test_image(net, test_depth, test_color, device, epoch, out_folder="./eval_test")
        print(f"Saved image for epoch {epoch}")


if __name__ == "__main__":
    main()
