from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import load_image, extract_usable_images
from tqdm import tqdm

class LytroDataset(Dataset):
    STRIDE = 16
    TRAIN = 60

    def __init__(self, lightFieldPaths: list[str], training=False):
        self.training = training
        self.lightFields = [extract_usable_images(load_image(path)) for path in tqdm(lightFieldPaths)]

    def __len__(self):
        if self.training:
            imgX, imgY = self.lightFields[0].shape[3:5]
            xSize = (imgX - self.TRAIN + 1) // self.STRIDE
            ySize = (imgY - self.TRAIN + 1) // self.STRIDE
            return len(self.lightFields) * 8 * 8 * xSize * ySize
        else:
            return len(self.lightFields) * 8 * 8

    def __getitem__(self, idx):
        """
        Must return a Width x Height x Channels

        return X, y
        X = (RGBUV * 4 + U'V') x W x H
        y = (RGB) x (W-p) x (H-p)
        where p = padding (6)

        U is downwards, V is to the right with a top left origin
        """

        # Output padding lost per side
        # single cnn padding
        p = 6
        # dual cnn padding
        # p = 12

        targetX = idx % 8
        targetY = (idx // 8) % 8

        if self.training:
            imgX, imgY = self.lightFields[0].shape[3:5]
            # patch size
            xSize = (imgX - self.TRAIN + 1) // self.STRIDE
            ySize = (imgY - self.TRAIN + 1) // self.STRIDE
            
            patchX = (idx // 8 // 8) % xSize
            patchY = (idx // 8 // 8 // xSize) % ySize

            field = (idx // 8 // 8 // xSize // ySize)

            if field > len(self.lightFields):
                raise Exception(f"well crap, you got too many fields {field}")

            outsize = self.TRAIN - p

            uvSize = (1, self.TRAIN, self.TRAIN)

            xStart = patchX * self.STRIDE
            yStart = patchY * self.STRIDE
            
            source = np.vstack((
                # Top left (0, 0)
                self.lightFields[field][0, 0, :, xStart:xStart+self.TRAIN, yStart:yStart+self.TRAIN],
                # Top left U, V
                np.full(uvSize, 0, dtype=np.float16),
                np.full(uvSize, 0, dtype=np.float16),
                # Top right (0, 1)
                self.lightFields[field][0, 7, :, xStart:xStart+self.TRAIN, yStart:yStart+self.TRAIN],
                # Top right U, V
                np.full(uvSize, 0, dtype=np.float16),
                np.full(uvSize, 1, dtype=np.float16),
                # Bottom left (1, 0)
                self.lightFields[field][7, 0, :, xStart:xStart+self.TRAIN, yStart:yStart+self.TRAIN],
                # Bottom left U, V
                np.full(uvSize, 1, dtype=np.float16),
                np.full(uvSize, 0, dtype=np.float16),
                # Bottom right (1, 1)
                self.lightFields[field][7, 7, :, xStart:xStart+self.TRAIN, yStart:yStart+self.TRAIN],
                # Bottom right U, V
                np.full(uvSize, 1, dtype=np.float16),
                np.full(uvSize, 1, dtype=np.float16),
                # U', V'
                np.full(uvSize, targetX / 7, dtype=np.float16),
                np.full(uvSize, targetY / 7, dtype=np.float16)
            ))
            target = self.lightFields[field][targetX, targetY, :, xStart+p:xStart+outsize, yStart+p:yStart+outsize]
            
            return source, target
        else:
            imgShape = (1,) + self.lightFields[0].shape[3:5]
            field = (idx // 8 // 8)

            if field > len(self.lightFields):
                raise Exception(f"well crap, you got too many fields {field}")
            
            source = np.vstack((
                # Top left (0, 0)
                self.lightFields[field][0, 0],
                # Top left U, V
                np.full(imgShape, 0, dtype=np.float16),
                np.full(imgShape, 0, dtype=np.float16),
                # Top right (0, 1)
                self.lightFields[field][0, 7],
                # Top right U, V
                np.full(imgShape, 0, dtype=np.float16),
                np.full(imgShape, 1, dtype=np.float16),
                # Bottom left (1, 0)
                self.lightFields[field][7, 0],
                # Bottom left U, V
                np.full(imgShape, 1, dtype=np.float16),
                np.full(imgShape, 0, dtype=np.float16),
                # Bottom right (1, 1)
                self.lightFields[field][7, 7],
                # Bottom right U, V
                np.full(imgShape, 1, dtype=np.float16),
                np.full(imgShape, 1, dtype=np.float16),
                # U', V'
                np.full(imgShape, targetX / 7, dtype=np.float16),
                np.full(imgShape, targetY / 7, dtype=np.float16)
            ))
            target = self.lightFields[field][targetX, targetY, :, p:-p, p:-p]
            
            return source, target
