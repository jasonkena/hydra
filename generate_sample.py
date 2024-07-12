import numpy as np
import random
import os
import glob
import h5py

# from scipy import ndimage
# import skimage
# from skimage.color import label2rgb
import cc3d
from tqdm import tqdm
import skimage


def main():
    base_path = "/data/projects/weilab/dataset/hydra/small_vesicle"
    files = sorted(glob.glob(os.path.join(base_path, "*.h5")))

    images = [x for x in files if "_im" in x]
    labels = [x for x in files if "_vesicle_small" in x]
    # (H, W, C)
    patch_size = (16, 16, 4)

    for im, lb in zip(images, labels):
        assert im.replace("_im", "") == lb.replace("_vesicle_small", "")

    patches = []
    for im, lb in tqdm(zip(images, labels), total=len(images)):
        with h5py.File(im, "r") as f:
            image = f["main"][:]
        with h5py.File(lb, "r") as f:
            label = f["main"][:]

        patches.extend(extract_patches(image, label, patch_size))
    # [N, C, H, W]
    patches = np.stack(patches, axis=0)
    np.save("patches.npy", patches)

    return patches


def extract_patches(image, label, patch_size, connectivity=6, dust_thresh=100):
    """
    Generate N x H x W x C patches from the given image and label
    Transpose is done on dimensions

    Parameters
    ----------
    image : C x H x W uint8
        [TODO:description]
    label : C x H x W bool
        [TODO:description]
    patch_size : (h, w, c) int
    connectivity : for cc3d
    dust_thresh : ignore small connected components

    returns 
    """
    cc = cc3d.connected_components(label, connectivity=connectivity)

    stats = cc3d.statistics(cc)
    num_segs = len(stats["voxel_counts"])

    patches = []

    # ignore BG
    for i in range(1, num_segs):
        if stats["voxel_counts"][i] < dust_thresh:
            continue
        z, y, x = stats["bounding_boxes"][i]
        patch = image[z.start : z.stop, y.start : y.stop, x.start : x.stop]
        # [C, H, W] -> [H, W, C]
        patch = patch.transpose(2, 0, 1)
        patch = skimage.transform.resize(patch, patch_size)
        patches.append(patch)

    return patches

def debug():
    base_path = "/data/projects/weilab/dataset/hydra/small_vesicle"
    files = sorted(glob.glob(os.path.join(base_path, "*.h5")))

    images = [x for x in files if "_im" in x]
    labels = [x for x in files if "_vesicle_small" in x]

    for im, lb in zip(images, labels):
        assert im.replace("_im", "") == lb.replace("_vesicle_small", "")

    patches = []
    for im, lb in tqdm(zip(images, labels), total=len(images)):
        id = os.path.basename(im).split("_")[0] # volN
        with h5py.File(im, "r") as f:
            image = f["main"][:]
        with h5py.File(lb, "r") as f:
            label = f["main"][:]
        file = h5py.File(f"{id}_sample.h5", "w")
        file.create_dataset("image", data=image)
        file.create_dataset("label", data=cc3d.connected_components(label, connectivity=6))
if __name__ == "__main__":
    main()
    # debug()
