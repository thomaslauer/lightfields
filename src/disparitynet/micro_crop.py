import imageio
import glob
import numpy as np
import pathlib
from tqdm import tqdm
from multiprocessing import Pool
import utils

INPUT = "../../datasets/general/raw/*.png"
OUTFOLDER = "../datasets/flowers_cropped"

def process(img):
    base = pathlib.Path(img).name
    out_name = f"{OUTFOLDER}/{base}"
    raw_data = utils.extract_usable_images(utils.load_image(img))
    data = utils.save_extracted(raw_data)
    imageio.imwrite(out_name, data)

utils.mkdirp(OUTFOLDER)

if __name__ == '__main__':
    files = glob.glob(INPUT)
    with Pool(4) as p:
        r = list(tqdm(p.imap(process, files), total=len(files)))
