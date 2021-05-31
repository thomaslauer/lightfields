import imageio
import glob
import numpy as np
import pathlib
from tqdm import tqdm
from multiprocessing import Pool
import utils
import params

# INPUT = "../../reference_implementation/Scenes/Rock.png"
# OUTFOLDER = "../datasets/microcropped_images"

INPUT = f"{params.drive_path}/datasets/raw/*.png"
OUTFOLDER = f"{params.drive_path}/datasets/microcropped_images"


def process(img):
    base = pathlib.Path(img).name
    out_name = f"{OUTFOLDER}/{base}"
    raw_data = utils.extract_usable_images(utils.load_image(img))
    data = utils.save_extracted(raw_data)
    imageio.imwrite(out_name, data)


utils.mkdirp(OUTFOLDER)

if __name__ == '__main__':
    files = glob.glob(INPUT)
    good_file_list = open("good.txt", "r")
    good = set(good_file_list.read().split("\n"))
    good_files = []
    for name in files:
        if pathlib.Path(name).name in good:
            good_files.append(name)
    with Pool(10) as p:
        r = list(tqdm(p.imap(process, good_files), total=len(good_files)))
