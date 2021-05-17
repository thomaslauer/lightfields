from sys import argv
import numpy as np
import imageio


def saveImg(img, offsetR, offsetC, filename):
    projection = img[offsetR::14, offsetC::14, :]
    imageio.imwrite(filename, projection, "PNG-FI")


if len(argv) < 3:
    print(f"python {argv[0]} <input_image> <output_folder>")

fullImg = imageio.imread(argv[1], "PNG-FI") / 2**16
outfolder = argv[2]

counter = 0
offset = 3
for r in range(8):
    for c in range(8):
        print(r, c)
        imgR = r + offset
        imgC = c + offset
        saveImg(fullImg, imgR, imgC, f"{outfolder}/0{r+1}_0{c+1}.png")
        counter += 1
