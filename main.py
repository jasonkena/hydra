import os
import glob
import numpy as np
import h5py

import pyroved as pv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import wandb
from reloading import reloading


# Add your custom dataset class here
class VesicleDataset(Dataset):
    def __init__(self, patch_file, transforms):
        self.data = np.load(patch_file)
        self.transforms = transforms
        # [N, H, W, C] -> [H, W, C]
        self.data_dim = self.data.shape[1:]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.float32) / 255
        img = torch.from_numpy(img)
        img = self.transforms(img)
        return (img,)


def dataset_with_indices(cls):
    # https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data = cls.__getitem__(self, index)
        assert isinstance(data, tuple)

        return data + (index,)

    return type(
        cls.__name__,
        (cls,),
        {
            "__getitem__": __getitem__,
        },
    )


def train(enable_wandb=True):
    # Initialize VAE model

    if enable_wandb:
        wandb.init(project="hydra")

    # Create a dataloader object

    train_dataset = VesicleDataset(
        patch_file="/data/adhinart/hydra/patches.npy", transforms=nn.Identity()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=32
    )
    rvae = pv.models.iVAE(
        train_dataset.data_dim[:2],
        extra_data_dim=train_dataset.data_dim[2],
        # latent_dim=16,
        latent_dim=2,
        invariances=None,
        # invariances=["r", "t"],
        dx_prior=0.5,
        dy_prior=0.5,
    )  # rotation and translation invariance
    # Initialize SVI trainer
    trainer = pv.trainers.SVItrainer(rvae)

    # Train for 100 epochs

    for _ in range(100):
    # for e in reloading(range(100)):
        trainer.step(train_loader)
        trainer.print_statistics()  # print running loss
        if enable_wandb:
            wandb.log({"loss": trainer.loss_history["training_loss"][-1]})
    rvae.save_weights("model")


if __name__ == "__main__":
    train()
