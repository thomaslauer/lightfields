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


def train(net, train_loader, optimizer, device, epoch):

    net.train()

    print(f"Training epoch {epoch}")
    for batch_data, target in tqdm(train_loader):
        optimizer.zero_grad()

        batch_data = batch_data.to(device)
        target = target.to(device)

        predicted = net(batch_data)
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
        for batch_data, target in tqdm(validation_loader):
            batch_data = batch_data.to(device)
            target = target.to(device)

            predicted = net(batch_data)
            loss = F.mse_loss(predicted, target)

            loss_sum += torch.sum(loss.cpu())

        # TODO: add other eval functions here

    return loss_sum



def main():

    # determine whether to use cuda or cpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # lightFieldPaths = [
    #     "../datasets/reflective_17_eslf.png",
    #     "../datasets/reflective_18_eslf.png",
    # ]

    # lightFieldPaths = glob.glob('../datasets/flowers_plants/raw/*_eslf.png')
    lightFieldPaths = glob.glob('../datasets/flowers_cropped/*.png')

    lightFieldPaths = lightFieldPaths[:len(lightFieldPaths)//2]
    # lightFieldPaths = lightFieldPaths[0:3]

    full_dataset = datasets.LytroDataset(lightFieldPaths, training=True, cropped=True)

    train_size = int(0.8 * len(full_dataset))
    validate_size = len(full_dataset) - train_size

    train_dataset, validate_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, validate_size]
    )


    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128, num_workers=4)

    validate_loader = DataLoader(validate_dataset, shuffle=True, batch_size=128, num_workers=4)

    net = networks.NaiveNet()

    if params.start_epoch != 0:
        # load previous epoch checkpoint
        net.load_state_dict(torch.load(utils.get_checkpoint_path(params.start_epoch-1)))

    # move net to cuda BEFORE setting optimizer variables
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=params.sgd_lr, momentum=params.sgd_momentum)

    for epoch in range(params.start_epoch, params.start_epoch + params.epochs):
        train(net, train_loader, optimizer, device, epoch)
        saveModel(net, epoch)
        loss = validate(net, validate_loader, device, epoch)
        print(f"Validation loss is {loss}")




if __name__ == "__main__":
    main()