import numpy as np
import imageio
import pathlib
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def get_checkpoint_path(epoch):
    return f"checkpoints/checkpoint_{epoch}.pth"


def get_idx_folder(folder, idx):
    return f"{folder}/{idx}"


def get_processed_patch_name(folder, idx, u, v, color=False):

    if color:
        return f"{get_idx_folder(folder, idx)}/{u}_{v}_color.npy"
    else:
        return f"{get_idx_folder(folder, idx)}/{u}_{v}_depth.npy"


def extract_usable_images(rawLightField: np.ndarray) -> np.ndarray:
    """Returns the usable section of the lightfield"""
    # img[r,c, ix,iy, 3]
    # Lytro images are 14x14 per subimage in linear RGBA
    # All sub areas should have an 8x8 sub region available which means 3px offset per side (8+3+3 = 14)
    # Additionally, slice to only RGB fields
    imgX = rawLightField.shape[0] // 14
    imgY = rawLightField.shape[1] // 14
    shape = (8, 8, imgX, imgY, 3)
    img = np.empty(shape, dtype=np.float16)
    offset = 3
    for r in range(8):
        for c in range(8):
            imgR = r + offset
            imgC = c + offset
            # moveaxis to make color the first index
            img[r, c, ...] = rawLightField[imgR::14, imgC::14, :3]
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
    h, w, c = data.shape
    return np.array(data.reshape((8, 8, h // 64, w, 3))).astype(np.float32) / 255


def save_extracted(data):
    _8, _8, h, w, _3 = data.shape
    return (data.reshape((h * 64, w, 3)) * 255).astype(np.uint8)


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
    return out
