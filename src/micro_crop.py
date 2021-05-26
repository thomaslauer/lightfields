import imageio
import glob
import numpy as np
import pathlib
from tqdm import tqdm
import utils

INPUT = "../datasets/flowers_plants/raw/*.png"
OUTFOLDER = "../datasets/flowers_cropped"

utils.mkdirp(OUTFOLDER)

for img in tqdm(glob.glob(INPUT)):
    base = pathlib.Path(img).name
    out_name = f"{OUTFOLDER}/{base}"
    raw_data = utils.extract_usable_images(utils.load_image(img))
    data = utils.save_extracted(raw_data)
    imageio.imwrite(out_name, data)
