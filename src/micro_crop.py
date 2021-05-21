import imageio
import glob
import numpy as np
import pathlib
from tqdm import tqdm
import utils

INPUT = "<input_glob>"
OUTFOLDER = "<output_folder>"

for img in tqdm(glob.glob(INPUT)):
    base = pathlib.Path(img).name
    out_name = f"{OUTFOLDER}/{base}"
    imageio.imwrite(out_name, (utils.extract_usable_images(utils.load_image(img)) * 255).astype(np.uint8))
