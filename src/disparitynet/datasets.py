import imageio
import numpy as np
import imageio
from tqdm import tqdm
from torch.utils.data import Dataset
import os

import params
import preprocess
import utils
from utils import load_image, extract_usable_images, load_extracted, mkdirp
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor

class Processor(object):
    def __init__(self, folder, lightField, grayField, patchSize, stride):
        self.patchSize = patchSize
        self.stride = stride
        self.lightField = lightField
        self.grayField = grayField
        self.folder = folder

    def __call__(self, coords):
        _8, _8, r, c, _3 = self.lightField.shape
        u, v = coords
        idx = 0
        for x_i, xStart in enumerate(range(0, r - self.patchSize, self.stride)):
            for y_i, yStart in enumerate(range(0, c - self.patchSize, self.stride)):
                mkdirp(utils.get_idx_folder(self.folder, idx))
                color_file_name = utils.get_processed_patch_name(self.folder, idx, u, v, color=True)
                if not os.path.exists(color_file_name):
                    colorPatch = self.lightField[u, v, xStart:xStart+self.patchSize, yStart:yStart+self.patchSize]
                    if colorPatch.shape != (self.patchSize, self.patchSize, 3):
                        raise Exception(f"incorrect size on color {idx}, {y_i}, {x_i}, {u}, {v}, got {colorPatch.shape}")
                    np.save(color_file_name, np.moveaxis(colorPatch, -1, 0).astype(np.float16))
                idx += 1

        depth = None
        idx = 0
        for x_i, xStart in enumerate(range(0, r - self.patchSize, self.stride)):
            for y_i, yStart in enumerate(range(0, c - self.patchSize, self.stride)):
                depth_file_name = utils.get_processed_patch_name(self.folder, idx, u, v, color=False)
                if not os.path.exists(depth_file_name):
                    if depth is None:
                        depth = self.depth(u, v)
                    depthPatch = depth[:, xStart:xStart+self.patchSize, yStart:yStart+self.patchSize]
                    if depthPatch.shape != (200, self.patchSize, self.patchSize):
                        raise Exception(f"incorrect size on depth {idx}, {y_i}, {x_i}, {u}, {v}, got {depthPatch.shape}")
                    np.save(depth_file_name, depthPatch.astype(np.float16))
                idx += 1

        return idx

    def depth(self, u, v):
        return preprocess.prepare_depth_features(self.grayField, u / 7, v / 7)

class LytroDataset(Dataset):
    STRIDE = 32
    TRAIN = 60
    TMP = f"{params.drive_path}/tmp2"

    def __init__(self, lightFieldPaths: list[str], training=False, cropped=False):
        self.training = training
        # NOTE: lightfields stored in 8x8xRxCxRGB
        def readImage(path):
            if cropped:
                return load_extracted(imageio.imread(path))
            else:
                return extract_usable_images(load_image(path))
        self.patches = []

        mkdirp(self.TMP)

        coords = np.moveaxis(np.mgrid[0:8, 0:8].reshape((2, -1)), 0, -1)

        for i, path in enumerate(lightFieldPaths):
            base = Path(path).stem

            imgDir = f"{self.TMP}/{base}"
            mkdirp(imgDir)
            print(f"({i}) Loading {path}")
            rawLightField = readImage(path)
            grayField = preprocess.crop_gray(rawLightField)

            with ProcessPoolExecutor(max_workers=6) as executor:
                results = list(tqdm(executor.map(Processor(imgDir, rawLightField, grayField, self.TRAIN, self.STRIDE), coords), total=len(coords)))

            for r in range(results[0]):
                self.patches.append((imgDir, r))

    def __len__(self):
        if self.training:
            return len(self.patches) * 64
        else:
            raise Exception("TODO: Fix eval.py")

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

        targetX = idx % 8
        targetY = (idx // 8) % 8
        u = targetX / 7
        v = targetY / 7
        patch = idx // 8 // 8

        if self.training:
            # Get the patch disparity
            folder, idx = self.patches[patch]
            depth = np.load(utils.get_processed_patch_name(folder, idx, targetX, targetY, color=False)).astype(np.float32)

            colorShape = (1, self.TRAIN - p*2, self.TRAIN - p*2)
            color = np.vstack((
                # Top left (0, 0)
                np.load(utils.get_processed_patch_name(folder, idx, 0, 0, color=True))[:, p:-p, p:-p].astype(np.float32),
                *self.uv(colorShape, 0, 0),
                # Top right (0, 1)
                np.load(utils.get_processed_patch_name(folder, idx, 0, 7, color=True))[:, p:-p, p:-p].astype(np.float32),
                *self.uv(colorShape, 0, 1),
                # Bottom left (1, 0)
                np.load(utils.get_processed_patch_name(folder, idx, 7, 0, color=True))[:, p:-p, p:-p].astype(np.float32),
                *self.uv(colorShape, 1, 0),
                # Bottom right (1, 1)
                np.load(utils.get_processed_patch_name(folder, idx, 7, 7, color=True))[:, p:-p, p:-p].astype(np.float32),
                *self.uv(colorShape, 1, 1),
                # U', V'
                np.full(colorShape, u, dtype=np.float32),
                np.full(colorShape, v, dtype=np.float32)
            ))

            target = np.load(utils.get_processed_patch_name(folder, idx, targetX, targetY, color=True))[:, p*2:-p*2, p*2:-p*2].astype(np.float32)

            return depth, color, target
        else:
            raise Exception("Ya done fucked up")
            field = (idx // 8 // 8)

            if field > len(self.lightFields):
                raise Exception(f"well crap, you got too many fields {field}")

            file_name = self.depthNames[field][targetX][targetY]
            depth = np.load(file_name)
            
            imgShape = (1, self.lightFields[field].shape[3] - p*2, self.lightFields[field].shape[4] - p*2)
            color = self.assemble_color_net(u, v, field, imgShape, p, p)
            p2 = p*2
            target = self.lightFields[field][targetX, targetY, :, p2:-p2, p2:-p2]
            return depth, color, target

    def uv(self, shape, u, v):
        return np.full(shape, u, dtype=np.float32), np.full(shape, v, dtype=np.float32)

    """ 
    def assemble_color_net(self, targetU, targetV, field, patchShape, xStart=0, yStart=0):
        _1, r, c = patchShape
        source = np.vstack((
            # Top left (0, 0)
            self.lightFields[field][0, 0, :, xStart:xStart+r, yStart:yStart+c],
            *self.uv(patchShape, 0, 0),
            # Top right (0, 1)
            self.lightFields[field][0, 7, :, xStart:xStart+r, yStart:yStart+c],
            *self.uv(patchShape, 0, 1),
            # Bottom left (1, 0)
            self.lightFields[field][7, 0, :, xStart:xStart+r, yStart:yStart+c],
            *self.uv(patchShape, 1, 0),
            # Bottom right (1, 1)
            self.lightFields[field][7, 7, :, xStart:xStart+r, yStart:yStart+c],
            *self.uv(patchShape, 1, 1),
            # U', V'
            np.full(patchShape, targetU, dtype=np.float32),
            np.full(patchShape, targetV, dtype=np.float32)
        ))
        return source
    """ 

