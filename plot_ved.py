import matplotlib

matplotlib.use("TkAgg")

import pyroved as pv
from main_ved import VesicleDataset

# from main import VesicleDataset, dataset_with_indices

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


def load():
    train_dataset = VesicleDataset(patch_file="patches.npy", transforms=nn.Identity())
    ved = pv.models.VED(
        input_dim=train_dataset.data_dim[:2],
        output_dim=train_dataset.data_dim[:2],
        input_channels=train_dataset.data_dim[2],
        output_channels=train_dataset.data_dim[2],
        # latent_dim=16,
        latent_dim=2,
    )  # rotation and translation invariance

    ved.load_weights("ved_model.pt")
    ved.eval()

    return ved, train_dataset


def infer(model, x):
    with torch.no_grad():
        # x: [11, 11]
        return model.encode(x)[0].squeeze()[-model.ndim :].tolist()


def get_embeddings(ved, train_dataset):
    embeddings = []
    for idx in tqdm(range(len(train_dataset))):
        img = train_dataset[idx][0]
        embeddings.append(infer(ved, img))

    np.save("embeddings.npy", embeddings)


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


def rescale_embeddings(embeddings):
    x_min, x_max = np.min(embeddings[:, 0]), np.max(embeddings[:, 0])
    y_min, y_max = np.min(embeddings[:, 1]), np.max(embeddings[:, 1])
    embeddings[:, 0] = (embeddings[:, 0] - x_min) / (x_max - x_min)
    embeddings[:, 1] = (embeddings[:, 1] - y_min) / (y_max - y_min)
    return embeddings


def project(volume):
    assert len(volume.shape) == 3
    # HWC
    return volume[volume.shape[0] // 2]
    # return volume[:, :, volume.shape[2] // 2]


def contrast(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def plot(train_dataset, filter=None, interactive=True):
    # interactive, whether to plot/activating onclick hook

    images = []

    for vol in train_dataset:
        images.append(project(vol[0].numpy()))

    embeddings = np.load("embeddings.npy")
    # embeddings = rescale_embeddings(
    #     embeddings
    # )  # hack because Datashader raises division by zero otherwise
    # plt.hist2d(embeddings[:, 0], embeddings[:, 1], bins=100)
    # plt.show()

    x1, x2, y1, y2 = (
        np.argmin(embeddings[:, 0]),
        np.argmax(embeddings[:, 0]),
        np.argmin(embeddings[:, 1]),
        np.argmax(embeddings[:, 1]),
    )

    # # NOTE: dumb hack to bypass broken datashader internals which raises shape error when labels is all one class?
    # # see https://github.com/holoviz/datashader/issues/1230
    # labels[x1] = 0
    # labels[x2] = 1

    extent = [
        embeddings[x1, 0],
        embeddings[x2, 0],
        embeddings[y1, 1],
        embeddings[y2, 1],
    ]

    print(f"extent: {extent}")
    # extent = _get_extent(embeddings)
    # fig_size = (800, 800)

    # get current axis
    ax = plt.gca()
    # ax.set_aspect("equal")
    ax.hist2d(
        embeddings[:, 0], embeddings[:, 1], bins=100, range=[extent[:2], extent[2:]]
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
        model = load()[0]
        fig = ax.get_figure()
        im = OffsetImage(
            np.concatenate(
                [contrast(images[0]), contrast(project(recons(model, 0, 0)))], axis=1
            ),
            zoom=5,
            cmap="gray",
        )
        # im = OffsetImage(images[0], zoom=5, cmap="gray")

        kd = KDTree(embeddings)
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
                        [contrast(images[idx]), contrast(project(recons(model, x, y)))],
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


if __name__ == "__main__":
    ved, train_dataset = load()
    get_embeddings(ved, train_dataset)

    # plt.imshow(project(train_dataset[0][0].numpy()), cmap="gray")
    # plt.show()
    embeddings = np.load("embeddings.npy")
    # plt.scatter(embeddings[:, 0], embeddings[:, 1])
    # plt.show()
    for i in range(len(embeddings)):
        # plot side by side
        fig, ax = plt.subplots(1, 2)
        image = train_dataset[i][0]
        inferred = infer(ved, image)
        print(inferred)
        ax[0].imshow(contrast(project(image.numpy())), cmap="gray")
        ax[1].imshow(
            contrast(project(recons(ved, inferred[0], inferred[1]))), cmap="gray"
        )
        plt.show()

    plot(train_dataset, interactive=True)

    #  setenv LD_LIBRARY_PATH LD_LIBRARY_PATH\:/data/adhinart/.conda/envs/vesicle/lib/
    # plot(interactive=False)
    # plt.show()

    # generate_all_figs()
