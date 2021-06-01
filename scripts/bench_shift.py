#!/usr/bin/env python

import timeit
import numpy as np
import cv2 as cv
from scipy import ndimage
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import imageio
from math import floor, ceil

# NOTE: U is down, V is right


def scipy_shift(img, output, u, v):
    ndimage.shift(img, [-u, -v], output=output, cval=np.nan, order=3, prefilter=False)


def scipy_shift_filter(img, output, u, v):
    ndimage.shift(img, [-u, -v], output=output, cval=np.nan, order=3, prefilter=True)


def opencv_affine(img, output, u, v):
    r, c = img.shape
    mat = np.float32([[1, 0, -v], [0, 1, -u]])
    cv.warpAffine(img, mat, (c, r), output, cv.INTER_CUBIC, cv.BORDER_REFLECT_101)
    if u < 0:
        output[:-floor(u)] = np.nan
    elif u > 0:
        output[-ceil(u):] = np.nan

    if v < 0:
        output[:, :-floor(v)] = np.nan
    elif v > 0:
        output[:, -ceil(v):] = np.nan


def make_caller(func, img, output, u, v):
    return lambda: func(img, output, u, v)


def stats(arr):
    return f"mean: {np.mean(arr):7.5}, median: {np.median(arr):7.5}"


def compare(name, img1, img2):
    img1 = np.nan_to_num(img1)
    img2 = np.nan_to_num(img2)
    ssim_f = structural_similarity(img1, img2, gaussian_weights=True)
    psnr_f = peak_signal_noise_ratio(img1, img2)
    ssim_b = structural_similarity(img2, img1, gaussian_weights=True)
    psnr_b = peak_signal_noise_ratio(img2, img1)
    print()
    print(f"Forards {name}")
    print(f"SSIM: {ssim_f}")
    print(f"PSNR: {psnr_f}")
    print(f"Backwards {name}")
    print(f"SSIM: {ssim_b}")
    print(f"PSNR: {psnr_b}")


def main():
    source = "../../tmp/flowers_25.png"
    img = cv.cvtColor(cv.imread(source, cv.IMREAD_COLOR)[
                      7::14, 7::14].astype(np.float32) / 255, cv.COLOR_BGR2GRAY)[:102, :102]
    print(img.shape)
    result_shift = np.empty(img.shape, dtype=np.float32)
    result_shift_filter = np.empty(img.shape, dtype=np.float32)
    result_affine = np.empty(img.shape, dtype=np.float32)

    count = 100
    u, v = -21, 10.98

    shifts = []
    shift_filters = []
    affines = []

    for i in range(3):
        print("Running shift")
        shift = timeit.Timer(make_caller(scipy_shift, img, result_shift, u, v)).timeit(count)
        shifts.append(shift)
        print(f"Took {shift}s")

        print("Running shift")
        shift = timeit.Timer(make_caller(scipy_shift_filter, img, result_shift_filter, u, v)).timeit(count)
        shift_filters.append(shift)
        print(f"Took {shift}s")

        print("Running affine")
        affine = timeit.Timer(make_caller(opencv_affine, img, result_affine, u, v)).timeit(count)
        affines.append(affine)
        print(f"Took {affine}s")

    print("Shift stats:")
    print(stats(shifts))
    print("Shift filter stats:")
    print(stats(shift_filters))
    print("Affine stats")
    print(stats(affines))

    _fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(img)
    axs[0, 1].imshow(result_affine)
    axs[1, 0].imshow(result_shift)
    axs[1, 1].imshow(result_shift_filter)
    plt.show()

    print(result_shift.shape)
    print(result_affine.shape)

    compare("scipy, cv", result_shift, result_affine)
    # compare("img, cv", img, result_affine)
    # compare("img, scipy", img, result_shift)
    # compare("img, scipy_filter", img, result_shift_filter)
    compare("scipy_filter, cv", result_shift_filter, result_affine)
    compare("scipy_filter, scipy", result_shift_filter, result_shift)


if __name__ == "__main__":
    main()
