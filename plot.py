import matplotlib

matplotlib.use("TkAgg")

import pyroved as pv
from main import VesicleDataset, dataset_with_indices, normalize

import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np
import imageio.v3 as iio
from tqdm import tqdm

import umap
import umap.plot
import matplotlib.pyplot as plt

import os
import glob
import h5py

from scipy.spatial import KDTree
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from tqdm import tqdm
from magicpickle import MagicPickle

from typing import Union
from pyroved.utils import generate_latent_grid, plot_img_grid, plot_spect_grid
import pyro.distributions as dist

# dir which contains all h5 files, umap.npy, and embeddings.pt
BASE_PATH = "/data/bccv/dataset/xiaomeng/mossy_terminal/ves"
LATENT_DIM = 2


def load(patch_data):
    train_dataset = VesicleDataset(patch_data, transforms=normalize)

    rvae = pv.models.iVAE(
        train_dataset.data_dim,
        # extra_data_dim=train_dataset.data_dim[2],
        latent_dim=LATENT_DIM,
        # invariances=None,
        invariances=["r", "t"],
        dx_prior=0.5,
        dy_prior=0.5,
    )  # rotation and translation invariance

    return rvae, train_dataset


def set_weights(model, weights):
    # move weights to device of model
    model.load_state_dict(weights)
    model.eval()
    return model


def infer(model, x):
    with torch.no_grad():
        # x: [11, 11]
        # https://github.com/ziatdinovmax/pyroVED/blob/7807ffb1cb415b3cc76c1e02d52465a8ae0eeae4/pyroved/models/ivae.py#L237
        # last 2 dims as latent dimensions
        mean, std = model.encode(x)
        mean, std = mean.squeeze(), std.squeeze()
        mean, std = mean[-LATENT_DIM:], std[-LATENT_DIM:]

        # [x mean, y mean, x std, y std]
        return mean.tolist() + std.tolist()


def get_embeddings(rvae, train_dataset):
    embeddings = []
    for idx in tqdm(range(len(train_dataset))):
        img = train_dataset[idx][0]
        embeddings.append(infer(rvae, img))
    # np.save("embeddings.npy", embeddings)
    return np.array(embeddings)


def recons(model, x, y):
    coord = torch.tensor([[x, y]]).float()
    img = model.decode(coord).squeeze(0).numpy()
    # img = (img * 255).astype(np.uint8)
    return img


# def recons(experiment, img):
#     recons = experiment.model.generate(img.unsqueeze(0), labels=0)
#     iio.imwrite(
#         "recons.png", (recons * 256).detach().numpy().reshape(16, 16).astype(np.uint8)
#     )
#
#     return recons


def _get_extent(points):
    """Compute bounds on a space with appropriate padding"""
    min_x = np.nanmin(points[:, 0])
    max_x = np.nanmax(points[:, 0])
    min_y = np.nanmin(points[:, 1])
    max_y = np.nanmax(points[:, 1])

    extent = (
        np.round(min_x - 0.05 * (max_x - min_x)),
        np.round(max_x + 0.05 * (max_x - min_x)),
        np.round(min_y - 0.05 * (max_y - min_y)),
        np.round(max_y + 0.05 * (max_y - min_y)),
    )

    return extent


def get_h5s():
    return sorted(glob.glob(os.path.join(BASE_PATH, "*_patch.h5")))


def rescale_embeddings(embeddings):
    x_min, x_max = np.min(embeddings[:, 0]), np.max(embeddings[:, 0])
    y_min, y_max = np.min(embeddings[:, 1]), np.max(embeddings[:, 1])
    embeddings[:, 0] = (embeddings[:, 0] - x_min) / (x_max - x_min)
    embeddings[:, 1] = (embeddings[:, 1] - y_min) / (y_max - y_min)
    return embeddings


def project(volume):
    assert len(volume.shape) == 3
    # HWC
    return volume[:, :, volume.shape[2] // 2]


def contrast(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def read_images(train_dataset):
    images = []
    for vol in train_dataset:
        images.append(vol[0].numpy())
    return images


def plot(model, images, embeddings, filter=None, interactive=True, std=False, bins=20):
    # interactive, whether to plot/activating onclick hook

    # embeddings = np.load("embeddings.npy")
    # __import__('pdb').set_trace()
    assert embeddings.shape[1] == 4
    # embeddings = rescale_embeddings(
    #     embeddings
    # )  # hack because Datashader raises division by zero otherwise
    # plt.hist2d(embeddings[:, 0], embeddings[:, 1], bins=100)
    # plt.show()

    if not std:
        x1, x2, y1, y2 = (
            np.argmin(embeddings[:, 0]),
            np.argmax(embeddings[:, 0]),
            np.argmin(embeddings[:, 1]),
            np.argmax(embeddings[:, 1]),
        )
    else:
        x1, x2, y1, y2 = (
            np.argmin(embeddings[:, 2]),
            np.argmax(embeddings[:, 2]),
            np.argmin(embeddings[:, 3]),
            np.argmax(embeddings[:, 3]),
        )

    # # NOTE: dumb hack to bypass broken datashader internals which raises shape error when labels is all one class?
    # # see https://github.com/holoviz/datashader/issues/1230
    # labels[x1] = 0
    # labels[x2] = 1

    if not std:
        extent = [
            embeddings[x1, 0],
            embeddings[x2, 0],
            embeddings[y1, 1],
            embeddings[y2, 1],
        ]
    else:
        extent = [
            embeddings[x1, 2],
            embeddings[x2, 2],
            embeddings[y1, 3],
            embeddings[y2, 3],
        ]

    print(f"extent: {extent}")
    # extent = _get_extent(embeddings)
    # fig_size = (800, 800)

    # get current axis
    ax = plt.gca()
    ax.set_aspect("equal")
    if not std:
        ax.hist2d(
            embeddings[:, 0],
            embeddings[:, 1],
            bins=bins,
            # range=[[-1.5, 1.5], [-1.5, 1.5]],
            range=[extent[:2], extent[2:]],
        )
    else:
        ax.hist2d(
            embeddings[:, 2],
            embeddings[:, 3],
            bins=bins,
            range=[extent[:2], extent[2:]],
        )
    # plt.show()

    # mapper = umap.UMAP()
    #
    # ax = umap.plot.points(
    #     mapper,
    #     points=embeddings,
    #     labels=labels,
    #     color_key_cmap="cool",
    #     subset_points=subset_points,
    # )

    if interactive:
        fig = ax.get_figure()
        im = OffsetImage(
            np.concatenate(
                [contrast(images[0]), contrast(recons(model, 0, 0))], axis=1
            ),
            zoom=5,
            cmap="gray",
        )
        # im = OffsetImage(images[0], zoom=5, cmap="gray")

        if not std:
            kd = KDTree(embeddings[:, :2])
        else:
            kd = KDTree(embeddings[:, 2:])
        xybox = (50.0, 50.0)
        ab = AnnotationBbox(
            im,
            (0, 0),
            xybox=xybox,
            xycoords="data",
            boxcoords="offset points",
            pad=0.3,
            arrowprops=dict(arrowstyle="->"),
        )
        # add it to the axes and make it invisible
        ax.add_artist(ab)
        ab.set_visible(False)

        def onclick(event):
            if event.inaxes == ax:
                x = event.xdata
                y = event.ydata
                print(f"real_x: {x}, real_y: {y}")

                dist, idx = kd.query([x, y])

                print(dist, idx)
                print(embeddings[idx])

                im.set_data(
                    np.concatenate(
                        [
                            contrast(images[idx]),
                            contrast(
                                recons(model, embeddings[idx, 0], embeddings[idx, 1])
                            ),
                        ],
                        axis=1,
                    )
                )

                w, h = fig.get_size_inches() * fig.dpi
                ws = (event.x > w / 2.0) * -1 + (event.x <= w / 2.0)
                hs = (event.y > h / 2.0) * -1 + (event.y <= h / 2.0)

                ab.xybox = (xybox[0] * ws, xybox[1] * hs)
                # make annotation box visible
                ab.set_visible(True)
                # place it at the position of the hovered scatter point
                ab.xy = (event.xdata, event.ydata)

                ab.set_visible(True)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()


# recons(experiment, test_img)

# test_img = iio.imread("../white.png").astype(np.float32) / 256
# test_img = torch.from_numpy(test_img).reshape(1, test_img.shape[0], test_img.shape[1])
#
# test_img = transforms.Resize((16,16))(test_img)
#
# experiment, data = load()
# get_embeddings(experiment, data)

# tsne()
# get_umap()


def generate_all_figs():
    plot(interactive=False)
    plt.savefig("all.png")

    for h5 in tqdm(get_h5s()):
        # clear figure
        plt.figure()
        terminal = os.path.basename(h5).split("_")[0]
        plot(filter=terminal, interactive=False)
        plt.savefig(f"{terminal}.png")


        
        

def custom_generate_latent_grid(d: int, **kwargs) -> torch.Tensor:
    """
    Generates a grid of latent space coordinates
    """
    if isinstance(d, int):
        d = [d, d]
    z_coord = kwargs.get("z_coord")
    if z_coord:
        z1, z2, z3, z4 = z_coord
        """ WRONG version
        grid_x = torch.linspace(z2, z1, d[0])
        grid_y = torch.linspace(z3, z4, d[1])
        """
        grid_x = torch.linspace(z1, z2, d[0])
        grid_y = torch.linspace(z4, z3, d[1])
    else:
        """ WRONG version
        grid_x = dist.Normal(0, 1).icdf(torch.linspace(0.95, 0.05, d[0]))
        grid_y = dist.Normal(0, 1).icdf(torch.linspace(0.05, 0.95, d[1]))
        """
        grid_x = dist.Normal(0, 1).icdf(torch.linspace(0.05, 0.95, d[0]))
        grid_y = dist.Normal(0, 1).icdf(torch.linspace(0.95, 0.05, d[1]))
    z = []
    """ WRONG version
    for xi in grid_x:
        for yi in grid_y:
    """
    for yi in grid_y:
        for xi in grid_x:
            z.append(torch.tensor([xi, yi]).float().unsqueeze(0))
    return torch.cat(z), (grid_x, grid_y)

def custom_manifold2d(self, d: int,
               y: torch.Tensor = None,
               plot: bool = True,
               **kwargs: Union[str, int, float]) -> torch.Tensor:
    """
    custom manifold with custom generate_latent_grid
    """
    z, (grid_x, grid_y) = custom_generate_latent_grid(d, **kwargs)

    z = [z]
    if self.c_dim > 0:
        if y is None:
            raise ValueError("To generate a manifold pass a conditional vector y") 
        y = y.unsqueeze(1) if 0 < y.ndim < 2 else y
        z = z + [y.expand(z[0].shape[0], *y.shape[1:])]
    loc = self.decode(*z, **kwargs)
    if plot:
        if self.ndim == 2:
            plot_img_grid(
                loc, d,
                extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                **kwargs)
        elif self.ndim == 1:
            plot_spect_grid(loc, d, **kwargs)
    return loc

def plot_1d(embeddings, vesicle_type_index):
    # plot 1D histogram of embeddings[:, vesicle_type_index]
    plt.hist(embeddings[:, vesicle_type_index], bins=20)
    plt.show()

import dill as pickle  # allow pickling of lambda functions

# https://stackoverflow.com/a/78399538/10702372
torch.serialization.register_package(0, lambda x: x.device.type, lambda x, _: x.cpu())

if __name__ == "__main__":
    with MagicPickle("think-jason") as mp:
        if mp.is_remote:
            patch_data = np.load("patches.npy")
            weights = torch.load("model.pt")
            rvae, train_dataset = load(patch_data)
            rvae = set_weights(rvae, weights)
            images = read_images(train_dataset)
            embeddings = get_embeddings(rvae, train_dataset)
            # move weights to cpu
            mp.save((weights, patch_data, images, embeddings))
        else:
            weights, patch_data, images, embeddings = mp.load()
            rvae, _ = load(patch_data)
            rvae = set_weights(rvae, weights)
            custom_manifold2d(rvae, d=12, cmap="gray")

            plot(rvae, images, embeddings, interactive=True, std=False)
            # plot(rvae, images, embeddings, interactive=True, std=True)

            
            # since horizontal axis (idx 0) differentiates vesicle type
            VESICLE_TYPE_INDEX = 0
            print(f"VESICLE_TYPE_INDEX: {VESICLE_TYPE_INDEX}, needs to be verified after every train")
            plot_1d(embeddings, VESICLE_TYPE_INDEX)

        # plt.imshow(project(train_dataset[0][0].numpy()), cmap="gray")
        # plt.show()
        # plt.scatter(embeddings[:, 0], embeddings[:, 1])
        # plt.show()
        # for i in range(len(embeddings)):
        #     # plot side by side
        #     fig, ax = plt.subplots(1, 2)
        #     ax[0].imshow(train_dataset[i][0].numpy(), cmap="gray")
        # #     __import__('pdb').set_trace()
        #     ax[1].imshow(recons(rvae, embeddings[i][0], embeddings[i][1]), cmap="gray")
        #     plt.show()
        #  setenv LD_LIBRARY_PATH LD_LIBRARY_PATH\:/data/adhinart/.conda/envs/vesicle/lib/
        # plot(interactive=False)
        # plt.show()

        # generate_all_figs()
