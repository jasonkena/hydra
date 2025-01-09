import re
import numpy as np
import random
import os
import glob
import h5py
import math
import imageio

# from scipy import ndimage
# import skimage
# from skimage.color import label2rgb
import cc3d
from tqdm import tqdm
import skimage

import sys

sys.path.append("/data/adhinart/em100um/utils")
from utils import pad_slice


def main():
    base_path = "/data/projects/weilab/dataset/hydra/small_vesicle"
    files = sorted(glob.glob(os.path.join(base_path, "*.h5")))
    files = [x for x in files if "vol3" not in x]  # skip test volume
    # files = [x for x in files if "vol4" not in x]  # skip test volume

    images = [x for x in files if "_im" in x]
    labels = [x for x in files if "_vesicle_small" in x]
    # (H, W, C)
    patch_size = (11, 11)

    for im, lb in zip(images, labels):
        assert im.replace("_im", "") == lb.replace("_vesicle_small", "")

    patches = []
    for im, lb in tqdm(zip(images, labels), total=len(images)):
        with h5py.File(im, "r") as f:
            image = f["main"][:]
        with h5py.File(lb, "r") as f:
            label = f["main"][:]

        patches.extend(extract_patches(image, label, patch_size))
    # [N, H, W]
    patches = np.stack(patches, axis=0)
    np.save("patches.npy", patches)
    print(patches.shape)

    return patches


def extract_patches(
    image, label, patch_size, connectivity=6, dust_thresh=0, extra_pad=5
):
    """
    Generate N x H x W patches from the given image and label

    Parameters
    ----------
    image : D x H x W uint8
        [TODO:description]
    label : D x H x W bool
        [TODO:description]
    patch_size : (h, w) int
    connectivity : for cc3d
    dust_thresh : ignore small connected components

    returns
    """
    patches = []
    for z in range(image.shape[0]):
        cc = cc3d.connected_components(label[z : z + 1], connectivity=connectivity)

        stats = cc3d.statistics(cc)
        num_segs = len(stats["voxel_counts"])
        print(f"num_segs: {num_segs}")

        # ignore BG
        for i in range(1, num_segs):
            if stats["voxel_counts"][i] < dust_thresh:
                continue
            _, y, x = stats["bounding_boxes"][i]
            # pad image using
            center = (int((y.start + y.stop) / 2), int((x.start + x.stop) / 2))
            max_size = max(y.stop - y.start, x.stop - x.start)
            patch = pad_slice(
                image,
                np.s_[
                    z : z + 1,
                    center[0]
                    - max_size // 2
                    - extra_pad : center[0]
                    + max_size // 2
                    + extra_pad,
                    center[1]
                    - max_size // 2
                    - extra_pad : center[1]
                    + max_size // 2
                    + extra_pad,
                ],
                mode="edge",
            )
            # now just [H, W]
            patch = patch.squeeze(0)
            assert patch.shape[0] == patch.shape[1]
            patch = skimage.transform.resize(patch, patch_size)
            patches.append(patch)

    return patches


def debug():
    base_path = "/data/projects/weilab/dataset/hydra/small_vesicle"
    files = sorted(glob.glob(os.path.join(base_path, "*.h5")))
    files = [x for x in files if "vol3" not in files]  # skip test volume

    images = [x for x in files if "_im" in x]
    labels = [x for x in files if "_vesicle_small" in x]

    for im, lb in zip(images, labels):
        assert im.replace("_im", "") == lb.replace("_vesicle_small", "")

    patches = []
    for im, lb in tqdm(zip(images, labels), total=len(images)):
        id = os.path.basename(im).split("_")[0]  # volN
        with h5py.File(im, "r") as f:
            image = f["main"][:]
        with h5py.File(lb, "r") as f:
            label = f["main"][:]
        file = h5py.File(f"{id}_sample.h5", "w")
        file.create_dataset("image", data=image)
        file.create_dataset(
            "label", data=cc3d.connected_components(label, connectivity=6)
        )

def new_main():
    # /data/adhinart/Downloads/NET_11_SV_241001.vsseg_export_s0199_Y0_X2.tif
    tifs = sorted(glob.glob("/data/adhinart/Downloads/*.tif"))
    # /data/adhinart/Downloads/hydra_export_s0199_Y0_X0.png
    pngs = sorted(glob.glob("/data/adhinart/Downloads/*.png"))

    tifs_dict = {}
    pngs_dict = {}

    pattern = r"s\d+_Y\d+_X\d+"
    for tif in tifs:
        code = re.search(pattern, tif).group()
        assert code not in tifs_dict
        tifs_dict[code] = tif
    for png in pngs:
        code = re.search(pattern, png).group()
        assert code not in pngs_dict
        pngs_dict[code] = png
    intersection = set(tifs_dict.keys()) & set(pngs_dict.keys())
    result_dict = {k: {"tif": tifs_dict[k], "png": pngs_dict[k]} for k in intersection}
    print(f"{len(result_dict)} tif-png pairs found")

    patch_size = (11, 11)
    patches = []
    for k, v in tqdm(result_dict.items()):
        # add new 0 axis
        label = imageio.imread(v["tif"])
        image = imageio.imread(v["png"])
        label = np.expand_dims(label, axis=0)
        image = np.expand_dims(image, axis=0)

        patches.extend(extract_patches(image, label, patch_size))
    patches = np.stack(patches, axis=0)
    np.save("patches.npy", patches)
    print(patches.shape)
    
    return patches

def new_new_main():
    # based on already preprocessed file
    file = h5py.File("/data/projects/weilab/dataset/hydra/results/vesicle_small_KR4_30-8-8_patch.h5")
    vol = file["key0"][:]
    assert vol.shape[-2:] == (11, 11)
    assert vol.shape[1] == 1
    patches = vol.squeeze(1)
    print(patches.shape)
    np.save("patches.npy", patches)

def new_new_new_main():
    # /data/projects/weilab/dataset/hydra/results/vesicle_small_*_30-8-8_patch.h5
    files = sorted(glob.glob("/data/projects/weilab/dataset/hydra/results/vesicle_small_*_30-8-8_patch.h5"))
    print(f"Found {len(files)} files")
    patches = []
    for file in files:
        file = h5py.File(file, "r")
        vol = file["key0"][:]
        assert vol.shape[-2:] == (11, 11)
        assert vol.shape[1] == 1
        patches.extend(vol.squeeze(1))
    patches = [x for x in patches if np.sum(x) > 0] # not all 0
    patches = np.stack(patches, axis=0)
    print(patches.shape)
    np.save("patches.npy", patches)


if __name__ == "__main__":
    new_new_new_main()
    # new_new_main()
    # new_main()
    # main()
    # debug()
