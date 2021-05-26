import numpy as np
import imageio
import pathlib


def get_checkpoint_path(epoch):
    return f"checkpoints/checkpoint_{epoch}.pth"


def extract_usable_images(rawLightField: np.ndarray) -> np.ndarray:
    """Returns the usable section of the lightfield"""
    # img[r,c, C, ix,iy]
    # Lytro images are 14x14 per subimage in linear RGBA
    # All sub areas should have an 8x8 sub region available which means 3px offset per side (8+3+3 = 14)
    # Additionally, slice to only RGB fields
    imgX = rawLightField.shape[0] // 14
    imgY = rawLightField.shape[1] // 14
    shape = (8, 8, 3, imgX, imgY)
    img = np.empty(shape, dtype=np.float16)
    offset = 3
    for r in range(8):
        for c in range(8):
            imgR = r + offset
            imgC = c + offset
            # moveaxis to make color the first index
            img[r, c, ...] = np2torch_color(rawLightField[imgR::14, imgC::14, :3])
    return img


def load_image(path):
    img = imageio.imread(path)
    bitdepth = 16 if img.dtype == np.uint16 else 8
    return img.astype(np.float32) / (2 ** bitdepth - 1)


def torch2np_color(img: np.ndarray) -> np.ndarray:
    """Converts from torch RGB x W x H to W x H x RGB"""
    return np.moveaxis(img, 0, -1)


def np2torch_color(img: np.ndarray) -> np.ndarray:
    """Converts from torch W x H x RGB to RGB x W x H"""
    return np.moveaxis(img, -1, 0)


def mkdirp(directory):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def load_extracted(data):
    h, w = data.shape
    return np.array(data.reshape((8, 8, 3, h // 8, w // 24))).astype(np.float32) / 255


def save_extracted(data):
    _8, _8, _3, h, w = data.shape
    return (data.reshape((h * 8, w * 3 * 8)) * 255).astype(np.uint8)
