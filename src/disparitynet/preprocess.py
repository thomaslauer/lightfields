import numpy as np
from scipy import interpolate

depth_resolution = 100
delta_disparity = 21

# Takes in a lightfield image of format 8x8xRxCx3
# u is down, v is right
def prepare_depth_features(LF, u, v):
    _8, _8, r, c, _3 = LF.shape
    grayLF = np.dot(LF, [0.299, 0.587, 0.114])

    features_stack = np.zeros(r, c, depth_resolution * 2, dtype=np.float32)

    x_view, y_view = (g.flatten() for g in np.mgrid[0:2, 0:2] - [u, v])

    count = 0
    for ind_depth, cur_depth in enumerate(np.linspace(-delta_disparity, delta_disparity, depth_resolution)):
        sheared_LF = np.zeros(4, r, c)
        X, Y = np.mgrid[0:r, 0:c]

        idx = 0
        for iax in [0, 7]:
            for iay in [0, 7]:
                curY = Y + cur_depth * y_view[idx]
                curX = X + cur_depth * x_view[idx]
                points = np.array(X, Y).reshape((-1, 2))
                sheared_LF[idx] = interpolate.griddata(points, grayLF[iax, iay], (curX, curY), method='cubic')
                idx += 1

        features_stack[..., ind_depth] = defocus_response(sheared_LF)
        features_stack[..., depth_resolution + ind_depth] = corresp_response(sheared_LF)
    return features_stack


def defocus_response(field):
    return np.nan_to_num(np.nanmean(field, axis=0))

def corresp_response(field):
    return np.nan_to_num(np.nanstd(field, axis=0))
