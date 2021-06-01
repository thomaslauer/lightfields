import numpy as np
import imageio
import pathlib
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2 as cv


def get_checkpoint_path(epoch, type="model"):
    return f"checkpoints/checkpoint_{epoch}_{type}.pth"


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
    img = np.empty(shape, dtype=np.float32)
    offset = 3
    for r in range(8):
        for c in range(8):
            imgR = r + offset
            imgC = c + offset
            img[r, c, ...] = rawLightField[imgR::14, imgC::14, :3]
    return img


def load_image(path):
    img = cv.imread(path, cv.IMREAD_COLOR | cv.IMREAD_ANYDEPTH)
    bitdepth = 16 if img.dtype == np.uint16 else 8
    return cv.cvtColor(np.divide(img, (2 ** bitdepth - 1), dtype=np.float32), cv.COLOR_BGR2RGB)


def torch2np_color(img: np.ndarray) -> np.ndarray:
    """Converts from torch RGB x W x H to W x H x RGB"""
    return np.moveaxis(img, 0, -1)


def np2torch_color(img: np.ndarray) -> np.ndarray:
    """Converts from torch W x H x RGB to RGB x W x H"""
    return np.moveaxis(img, -1, 0)


def mkdirp(directory):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def load_cropped_path(path):
    data = cv.cvtColor(cv.imread(path, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    h, w, _3 = data.shape
    return np.divide(data, 255., dtype=np.float32).reshape(8, 8, h // 64, w, 3)


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
    return np.clip(out, 0, 1)


def save_disparity(name, disparity):
    save_image(name, disparity, (np.min(disparity), np.max(disparity)))


def save_image(name, img, minmax=(0, 1)):
    """Takes an image on the scale"""
    clipped = (np.clip(img, minmax[0], minmax[1]) - minmax[0]) * (255 / (minmax[1] - minmax[0]))
    imageio.imwrite(name, clipped.astype(np.uint8))

def stack_warps(warp):
    _12, r, c = warp.shape
    corners = np.moveaxis(warp.reshape(4, 3, r, c), 1, -1)

    warped = np.vstack((
        np.hstack((corners[0], corners[1])),
        np.hstack((corners[2], corners[3]))))
    return warped
