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
    # enable cudnn
    torch.backends.cudnn.benchmark = True

    net.train()

    loss_sum = 0

    print(f"Training epoch {epoch}")
    for i, (depth, color, target) in enumerate(tqdm(train_loader)):
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
        loss_sum += torch.sum(loss.cpu()).item()

        if (i + 1) % 500 == 0:
            # TODO: THOMAS FIX!!!
            append_loss("loss/loss_train.csv", epoch, i, loss_sum / 500)
            loss_sum = 0


def saveModel(net, epoch):
    utils.mkdirp("./checkpoints")
    torch.save(net.state_dict(), utils.get_checkpoint_path(epoch))


def append_loss(file, epoch, batch, loss):
    with open(file, "a") as myfile:
        myfile.write(f"{epoch},{batch},{loss}\n")


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
    append_loss("loss/validation.csv", epoch, 0, loss_sum)

    return loss_sum


def test_image(net, depth, color, target, device, epoch, out_folder="./eval_test"):
    # disable cudnn
    torch.backends.cudnn.benchmark = False

    utils.mkdirp(out_folder)
    net.eval()
    with torch.no_grad():
        depth = torch.unsqueeze(torch.Tensor(depth), 0).to(device)
        color = torch.unsqueeze(torch.Tensor(color), 0).to(device)

        images = color[:, :-2, :, :]
        novelLocation = color[:, -2:, :, :]

        disp, warp, output = net.all_steps(depth, images, novelLocation)

        output = output[0].cpu()
        disp = disp[0, 0].cpu().numpy()
        warp = warp[0, :12].cpu().numpy()
        _12, r, c = warp.shape
        corners = np.moveaxis(warp.reshape(4, 3, r, c), 1, -1)

        warped = np.vstack((
            np.hstack((corners[0], corners[1])),
            np.hstack((corners[2], corners[3]))))

        # TODO: compute loss against target
        loss = torch.sum(F.mse_loss(output, torch.tensor(target)).detach().cpu())
        append_loss("loss/ref_image.csv", epoch, 0, loss)

        print(f"Full image loss is {loss:8.7}")
        img = utils.torch2np_color(output.detach().numpy())
        imageio.imwrite(f"{out_folder}/epoch_{epoch:03}_result.png", utils.adjust_tone(img))
        imageio.imwrite(f"{out_folder}/epoch_{epoch:03}_warped.png", utils.adjust_tone(warped))
        disp_min = disp.min()
        disp_max = disp.max()
        print(f"Disparity: minmax: [{disp_min}, {disp_max}], std: {np.std(disp)}, mean: {np.mean(disp)}")
        imageio.imwrite(f"{out_folder}/epoch_{epoch:03}_disparity.png", (disp - disp_min) / (disp_max - disp_min))


def main():
    utils.mkdirp("loss")

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
    test_lightfield = f'{params.drive_path}/datasets/microcropped_images/flowers_plants_25_eslf.png'

    # Fetch test data
    if params.run_test:
        print("Building test data")
        test_data = datasets.LytroDataset([test_lightfield], training=False, cropped=True)
        test_depth, test_color, test_target = test_data[36]
    # clear memory if possible...
    test_data = None
    print("Splitting train/validate")

    train_size = int(0.9 * len(full_dataset))
    validate_size = len(full_dataset) - train_size

    train_dataset, validate_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, validate_size], generator=torch.Generator().manual_seed(42)
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
    # optimizer = torch.optim.SGD(net.parameters(), lr=params.sgd_lr, momentum=params.sgd_momentum)
    optimizer = torch.optim.Adam(net.parameters(), lr=params.adam_lr)
    if params.start_epoch != 0 and params.run_test:
        test_image(net, test_depth, test_color, test_target, device, params.start_epoch - 1, out_folder="./eval_test")

    for epoch in range(params.start_epoch, params.start_epoch + params.epochs):
        train(net, train_loader, optimizer, device, epoch)
        saveModel(net, epoch)
        loss = validate(net, validate_loader, device, epoch)
        print(f"Validation loss is {loss}")
        if params.run_test:
            test_image(net, test_depth, test_color, test_target, device, epoch, out_folder="./eval_test")
            print(f"Saved image for epoch {epoch}")


def only_test_image():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Building training data")
    flower_9 = f'../datasets/people_cropped/people_6_eslf.png'

    print("Building test data")
    # Fetch test data
    test_data = datasets.LytroDataset([flower_9], training=False, cropped=True)
    test_depth, test_color, test_target = test_data[36]
    net = networks.FullNet(device)

    if params.start_epoch != 0:
        # load previous epoch checkpoint
        net.load_state_dict(torch.load(utils.get_checkpoint_path(
            params.start_epoch-1), map_location=torch.device(device)))

    # move net to cuda BEFORE setting optimizer variables
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=params.sgd_lr, momentum=params.sgd_momentum)
    if params.start_epoch != 0:
        test_image(net, test_depth, test_color, test_target, device, params.start_epoch - 1, out_folder="./eval_test")


if __name__ == "__main__":
    # only_test_image()
    main()
