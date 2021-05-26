import numpy as np

depth_resolution = 100
delta_disparity = 21

# Takes in a lightfield image of format 8x8xRxCx3
# u is down, v is right
def prepare_depth_features(LF, u, v):
    _8, _8, r, c, _3 = LF.size
    grayLF = np.dot(LF, [0.299, 0.587, 0.114])

    defocus_stack = np.zeros(r, c, depth_resolution, dtype=np.float32)
    corresp_stack = np.zeros(r, c, depth_resolution, dtype=np.float32)
    features_stack = np.zeros(r, c, depth_resolution * 2, dtype=np.float32)

    ind_depth = 0
    count = 0
    for cur_depth in np.linspace(-delta_disparity, delta_disparity, depth_resolution):
        sheared_LF = np.zeros(r, c, 64)
        X, Y = np.mgrid[0:c, 0:r]

        view = 0
        for iax in range(8):
            for iay in range(8):
                curY = Y + cur_depth * 
                curX = X + cur_depth * 
