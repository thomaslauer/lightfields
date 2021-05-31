import numpy as np
import imageio
import matplotlib.pyplot as plt


def showImg(img, offsetR, offsetC, filename):
    projection = img[offsetR::14, offsetC::14, :]
    imageio.imwrite(filename, projection, "PNG-FI")


fullImg = imageio.imread("reflective_29_eslf.png")

counter = 0
for r in range(14):
    for c in range(14):
        showImg(fullImg, r, c, f"output/{counter}.png")
        counter += 1
