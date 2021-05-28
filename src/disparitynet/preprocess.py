import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import imageio
import utils
from tqdm import tqdm

depth_resolution = 100
delta_disparity = 21
DEBUG = False

# Takes in a lightfield image of format 8x8xRxCx3
# u is down, v is right


def prepare_depth_features(LF, u, v):
    _8, _8, r, c, _3 = LF.shape
    grayLF = np.dot(LF, [0.299, 0.587, 0.114])

    features_stack = np.zeros((r, c, depth_resolution * 2), dtype=np.float32)

    x_view, y_view = (g.flatten() - sub for g, sub in zip(np.mgrid[0:2, 0:2], [u, v]))

    for ind_depth, cur_depth in enumerate(tqdm(np.linspace(-delta_disparity, delta_disparity, depth_resolution))):
        sheared_LF = np.empty((4, r, c), dtype=np.float32)
        X, Y = np.mgrid[0:r, 0:c]

        # idx = 0
        # for iax in [0, 7]:
        #     for iay in [0, 7]:
        #         curY = Y + cur_depth * y_view[idx]
        #         curX = X + cur_depth * x_view[idx]
        #         points = np.array([X, Y]).reshape((-1, 2))
        #         sheared_LF[idx] = interpolate.griddata(points, grayLF[iax, iay].flatten(), (curX, curY), method='cubic')
        #         idx += 1

        idx = 0
        for iax in [0, 7]:
            for iay in [0, 7]:
                shiftX = cur_depth * x_view[idx]
                shiftY = cur_depth * y_view[idx]
                ndimage.shift(grayLF[iax, iay], [-shiftX, -shiftY], output=sheared_LF[idx], cval=np.nan)
                idx += 1

        if DEBUG:
            fig, axs = plt.subplots(2, 2)
            for x, img in zip(axs.flatten(), sheared_LF):
                x.imshow(img)
            plt.show()

        features_stack[..., ind_depth] = defocus_response(sheared_LF)
        features_stack[..., depth_resolution + ind_depth] = corresp_response(sheared_LF)
    return features_stack


def defocus_response(field):
    return np.nan_to_num(np.nanmean(field, axis=0))


def corresp_response(field):
    return np.nan_to_num(np.nanstd(field, axis=0))


def test(w=200, h=300):
    out = prepare_depth_features(np.random.rand(8, 8, w, h, 3).astype(np.float32), 0.5, 0)
    print(np.count_nonzero(out))
    return out


if __name__ == "__main__":
    # test()
    prepare_depth_features(utils.load_extracted(imageio.imread(
        "../datasets/people_cropped/people_6_eslf.png")), 1, 1)
