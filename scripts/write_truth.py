#!/usr/bin/env python

import imageio
import numpy as np
import sys
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def adjust_tone(img):
    """
    Use the same tone adjustment as the ref impl. Lytro saves very bad saturation
    in raw format, so bump it UPPPP!!!
    """
    out = np.clip(img, 0, 1)
    out = out ** (1 / 1.5)
    out = rgb_to_hsv(out)
    out[..., 1] *= 1.5
    out = hsv_to_rgb(out)
    return np.clip(out, 0, 1)


# image to extract in range [0, 7]. Will be X -> Right, Y -> Down
coords = (4, 4)

source = sys.argv[1]

startR = coords[1] + 3
startC = coords[0] + 3

p = 12

truth = imageio.imread(source)[startR::14, startC::14, :3][p:-p, p:-p]

imageio.imwrite("../final_paper/images/rocks/truth_05_05.png", adjust_tone(truth.astype(np.float32)/255))
