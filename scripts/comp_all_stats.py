#!/usr/bin/env python

import sys
import cv2 as cv
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from pathlib import Path


# ours   truth
# pos    theirs


truth = sys.argv[1]
ours = sys.argv[2]
theirs = sys.argv[3]

truth_image = cv.imread(truth, cv.IMREAD_COLOR)

idx = 1

p = 12 + 10

def adjust_tone(img):
    """
    Use the same tone adjustment as the ref impl. Lytro saves very bad saturation
    in raw format, so bump it UPPPP!!!
    """
    out = np.clip(img.astype(np.float32) / 255, 0, 1)
    out = out ** (1 / 1.5)
    out = rgb_to_hsv(out)
    out[..., 1] *= 1.5
    out = (hsv_to_rgb(out) * 255)
    return np.clip(out, 0, 255).astype(np.uint8)

def get_truth(v, u):
    startR = u + 2
    startC = v + 2
    return truth_image[startR::14, startC::14][p:-p, p:-p]

def get_ours(u, v):
    path = f"{ours}/nn_{v:02}_{u:02}.png"
    # print(path)
    return cv.imread(path, cv.IMREAD_COLOR)[10:-10, 10:-10]

def get_theirs(u, v, shape):
    path = f"{theirs}/{v:02}_{u:02}.png"
    # print(path)
    return cv.imread(path, cv.IMREAD_COLOR)[:shape[0], :shape[1]]

def get_pos(v, u, shape):
    arr = np.zeros(shape, dtype=np.uint8)
    padding = (shape[1] - shape[0]) // 2
    size = shape[0] // 8
    arr[:, :padding] = 255
    arr[:, -padding:] = 255
    uS = (u - 1) * size
    vS = (v - 1) * size + padding
    arr[uS:uS+size, vS:vS+size, 2] = 255
    return arr

def compare():
    print("img,u,v,our_ssim,their_ssim,our_psnr,their_psnr")
    for u in range(1, 9):
        for v in range(1, 9):
            ground = get_truth(u, v)
            our_img = get_ours(u, v)
            their_img = get_theirs(u, v, ground.shape)

            # plt.imshow(their_img)
            # plt.show()
            # plt.imshow(ground)
            # plt.show()
            # plt.imshow(our_img)
            # plt.show()
            # exit()

            our_ssim = structural_similarity(ground, our_img, gaussian_weights=True, multichannel=True)
            our_psnr = peak_signal_noise_ratio(ground, our_img)
            their_ssim = structural_similarity(ground, their_img, gaussian_weights=True, multichannel=True)
            their_psnr = peak_signal_noise_ratio(ground, their_img)

            print(f"{Path(truth).stem},{u},{v},{our_ssim},{their_ssim},{our_psnr},{their_psnr}")
            sys.stdout.flush()

if __name__ == "__main__":
    compare()
