import os
import torch
import h5py
import numpy as np
import skimage.transform
import cc3d
from typing import List, Union
import magicpickle as mp
from main import train
from plot import set_weights, load, read_images, get_embeddings


def main(sample_path):
    em = h5py.File(os.path.join(sample_path, "EM.h5"), "r")["main"][:]
    sv = h5py.File(os.path.join(sample_path, "SV.h5"), "r")["main"][:]

    patches = extract_patches(
        em,
        sv,
        patch_size=(11, 11),
    )
    patches = np.stack(patches, axis=0)
    print(f"Extracted {patches.shape[0]} patches of shape {patches.shape[1:]}")
    np.savez("demo_patches.npz", patches=patches)

    train("demo_patches.npz", enable_wandb=False, model_name="demo_model")
    rvae, train_dataset = load(patches)
    weights = torch.load("demo_model.pt")
    set_weights(rvae, weights)
    images = read_images(train_dataset)
    embeddings = get_embeddings(rvae, train_dataset)

    mp.send((weights, patches, images, embeddings, None))


def extract_patches(
    image, label, patch_size, connectivity=6, dust_thresh=0, extra_pad=5
):
    """
    Generate N x H x W patches from the given image and label
    """
    patches = []
    for z in range(image.shape[0]):
        cc = cc3d.connected_components(label[z : z + 1], connectivity=connectivity)

        stats = cc3d.statistics(cc)
        num_segs = len(stats["voxel_counts"])

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


def pad_slice(
    vol: np.ndarray, slices: List[Union[slice, int]], mode: str
) -> np.ndarray:
    """
    Given a n-dim volume (np-like array which supports np.s_ slicing) and a slice which may be out of bounds,
    zero-pad the volume to the dimensions of the slice

    the slices have to be in one of the following formats:
        - int (e.g. vol[0])
        - slice(None, None, None) (e.g. vol[:])
        - slice(start, stop, None) (e.g. vol[0:10]) -> the dimensions here will be padded

    output dimension will be
        - 1 if the slice is an int
        - (stop - start) if start and stop are not None
        - vol.shape[i] if start is None and stop is None

    notably, it does not handle ellipsis or np.newaxis

    Parameters
    ----------
    vol
    slices
    mode: np.pad mode
    """
    assert len(vol.shape) == len(
        slices
    ), f"Volume and slices must have the same number of dimensions, given {len(vol.shape)} and {len(slices)}"
    for i, s in enumerate(slices):
        if isinstance(s, int):
            continue
        else:
            assert isinstance(
                s, slice
            ), f"Slice must be an int or a slice, given {type(s)}"
            assert s.step is None, f"Slice step must be None, given {s.step}"
            assert (s.start is None) == (
                s.stop is None
            ), f"Slice start and stop must both be None or not None, given {s.start} and {s.stop}"
            if s.start is not None:
                assert (
                    s.start < s.stop
                ), f"Slice start must be less than stop, given {s.start} and {s.stop}"
                # NOTE: s.start is allowed to be negative
                assert (
                    s.start < vol.shape[i]
                ), f"Slice start must be less than volume shape, given {s.start} and {vol.shape[i]}"
                assert s.stop > 0, f"Slice stop must be greater than 0, given {s.stop}"

    output_shape = []
    for i, s in enumerate(slices):
        if isinstance(s, int):
            output_shape.append(1)
            assert (
                0 <= s < vol.shape[i]
            ), f"Slice {s} is out of bounds for dimension {i}, which has size {vol.shape[i]}"
        else:
            output_shape.append(
                s.stop - s.start if s.start is not None else vol.shape[i]
            )

    input_slices = []
    for i, s in enumerate(slices):
        if isinstance(s, int):
            input_slices.append(s)
        else:
            if s.start is None:
                input_slices.append(slice(None))
            else:
                input_slices.append(slice(max(0, s.start), min(vol.shape[i], s.stop)))

    pad_widths = []
    for i, s in enumerate(slices):
        if isinstance(s, int) or s.start is None:
            pad_widths.append((0, 0))
        else:
            pad_widths.append(
                (
                    max(0, -s.start),
                    max(0, s.stop - vol.shape[i]),
                )
            )

    # so if scalar is indexed i.e. np.arange(5)[0], shape will be [1] instead of ()
    output = np.zeros(output_shape, dtype=vol.dtype)
    output[:] = np.pad(vol[tuple(input_slices)], pad_widths, mode=mode)

    return output


if __name__ == "__main__":
    main("sample_data/V1")
