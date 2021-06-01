import numpy as np
import matplotlib.pyplot as plt
import imageio
import utils
import cv2 as cv
from scipy import ndimage
from math import floor, ceil

depth_resolution = 100
delta_disparity = 21
DEBUG = False

"""
Pipeline:
 <pre-crop> (out of line)
 init:
 Load images
   prepare_depth LF: 8*8*3*r*c*(4) ~192    disp: 200*r*c*(4)*64 ~12800
 Create patches (train only)
   np.save
 run: np.load
"""

# Takes in a lightfield image of format 8x8xRxCx3
# u is down, v is right


def crop_gray(LF):
    _8, _8, r, c, _3 = LF.shape
    grayLF = np.empty((2, 2, r, c), dtype=np.float32)
    cv.cvtColor(LF[0, 0], cv.COLOR_RGB2GRAY, grayLF[0, 0])
    cv.cvtColor(LF[0, 7], cv.COLOR_RGB2GRAY, grayLF[0, 1])
    cv.cvtColor(LF[7, 0], cv.COLOR_RGB2GRAY, grayLF[1, 0])
    cv.cvtColor(LF[7, 7], cv.COLOR_RGB2GRAY, grayLF[1, 1])
    return grayLF


def opencv_shift(img, output, u, v):
    r, c = img.shape
    mat = np.float32([[1, 0, -v], [0, 1, -u]])
    cv.warpAffine(img, mat, (c, r), output, cv.INTER_CUBIC, cv.BORDER_REFLECT_101)
    # NOTE: To correctly match the ndimage impl, we need to use cubic interp with border reflection
    # however, this causes the background that didn't exist to be filled with values. To fix this
    # we need to explicitly set those to NaN values. We can't just use border mode constant with nan
    # value because this causes accidental cropping :'(
    if u < 0:
        output[:-floor(u)] = np.nan
    elif u > 0:
        output[-ceil(u):] = np.nan

    if v < 0:
        output[:, :-floor(v)] = np.nan
    elif v > 0:
        output[:, -ceil(v):] = np.nan


def prepare_depth_features(grayLF, u, v):
    _2, _2, r, c = grayLF.shape

    features_stack = np.zeros((depth_resolution * 2, r, c), dtype=np.float32)

    x_view, y_view = (g.flatten() - sub for g, sub in zip(np.mgrid[0:2, 0:2], [u, v]))

    # print("Preparing depth features")
    for ind_depth, cur_depth in enumerate(np.linspace(-delta_disparity, delta_disparity, depth_resolution)):
        sheared_LF = np.empty((4, r, c), dtype=np.float32)

        idx = 0
        for iax in [0, 1]:
            for iay in [0, 1]:
                shiftX = cur_depth * x_view[idx]
                shiftY = cur_depth * y_view[idx]
                opencv_shift(grayLF[iax, iay], sheared_LF[idx], shiftX, shiftY)
                # ndimage.shift(grayLF[iax, iay], [-shiftX, -shiftY], output=sheared_LF[idx], cval=np.nan)
                idx += 1

        if DEBUG:
            fig, axs = plt.subplots(2, 2)
            for x, img in zip(axs.flatten(), sheared_LF):
                x.imshow(img)
            plt.show()

        features_stack[ind_depth] = defocus_response(sheared_LF)
        features_stack[depth_resolution + ind_depth] = corresp_response(sheared_LF)
    return features_stack


def defocus_response(field):
    return np.nan_to_num(np.nanmean(field, axis=0))


def corresp_response(field):
    return np.nan_to_num(np.nanstd(field, axis=0))


def test(w=200, h=300):
    out = prepare_depth_features(np.random.rand(8, 8, w, h, 3).astype(np.float32), 0.5, 0)
    print(np.count_nonzero(out))
    return out
