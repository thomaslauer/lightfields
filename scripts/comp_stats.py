#!/usr/bin/env python

import imageio
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import numpy as np
import sys


# image to extract in range [0, 7]. Will be X -> Right, Y -> Down
coords = (4, 4)

source, *raws = sys.argv[1:]

startR = coords[1] + 3
startC = coords[0] + 3

p = 12 + 10

truth = imageio.imread(source)[startR::14, startC::14, :3][p:-p, p:-p]
r, c, _3 = truth.shape

for rawPath in raws:
    raw = imageio.imread(rawPath)  # [:-1]
    R, C, _3 = raw.shape

    print(raw.shape, truth.shape)

    if R - r == 20 and C - c == 20:
        print("cropping our image")
        raw = raw[10:-10, 10:-10]

    ssim = structural_similarity(truth, raw, gaussian_weights=True, multichannel=True)
    psnr = peak_signal_noise_ratio(truth, raw)
    print(rawPath)
    print(f"SSIM: {ssim:.5}")
    print(f"PSNR: {psnr:.5}")
