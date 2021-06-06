#!/usr/bin/env python

import sys
import cv2 as cv
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# ours   truth
# pos    theirs


out_folder = sys.argv[1]
truth = sys.argv[2]
ours = sys.argv[3]
theirs = sys.argv[4]

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
    return adjust_tone(truth_image[startR::14, startC::14][p:-p, p:-p])

def get_ours(u, v):
    path = f"{ours}/nn_{v:02}_{u:02}.png"
    # print(path)
    return cv.imread(path, cv.IMREAD_COLOR)[10:-10, 10:-10]

def get_theirs(u, v, shape):
    return cv.imread(f"{theirs}/{v:02}_{u:02}.png", cv.IMREAD_COLOR)[:shape[0], :shape[1]]

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
    for v in range(1, 9):
        order = range(8, 0, -1) if v % 2 == 0 else range(1, 9)
        for u in order:
            print(u, v)
            ground = get_truth(u, v)
            our_img = get_ours(u, v)
            their_img = get_theirs(u, v, ground.shape)
            pos_img = get_pos(u, v, ground.shape)

            output = np.vstack((
                np.hstack((our_img, ground)),
                np.hstack((pos_img, their_img))
            ))

            cv.imwrite(f"{out_folder}/ffmpeg_{idx:03}.png", output)
            idx += 1

    for u in range(1, 9):
        order = range(8, 0, -1) if u % 2 == 0 else range(1, 9)
        for v in order:
            print(u, v)
            ground = get_truth(u, v)
            our_img = get_ours(u, v)
            their_img = get_theirs(u, v, ground.shape)
            pos_img = get_pos(u, v, ground.shape)

            output = np.vstack((
                np.hstack((our_img, ground)),
                np.hstack((pos_img, their_img))
            ))

            cv.imwrite(f"{out_folder}/ffmpeg_{idx:03}.png", output)
            idx += 1

if __name__ == "__main__":
    compare()
