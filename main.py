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
    def __init__(self, transforms):
        base_path = "/data/bccv/dataset/xiaomeng/mossy_terminal/ves"
        h5s = sorted(glob.glob(os.path.join(base_path, "*_patch.h5")))
        vols = []
        for h5 in h5s:
            with h5py.File(h5, "r") as f:
                vols.append(f["main"][:])
        vols = np.concatenate(vols, axis=0)
        self.vols = vols
        self.transforms = transforms

    def __len__(self):
        return self.vols.shape[0]

    def __getitem__(self, idx):
        img = self.vols[idx].astype(np.float32) / 255
        img = torch.from_numpy(img).reshape(img.shape[0], img.shape[1])
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


def train():
    # Initialize VAE model

    wandb.init(project="mossy-terminal")
    data_dim = (11, 11)

    rvae = pv.models.iVAE(
        data_dim, latent_dim=2, invariances=["r", "t"], dx_prior=0.5, dy_prior=0.5
    )  # rotation and translation invariance

    # Create a dataloader object

    train_dataset = VesicleDataset(transforms=nn.Identity())
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=32
    )
    # Initialize SVI trainer
    trainer = pv.trainers.SVItrainer(rvae)

    # Train for 100 epochs

    try:
        for e in reloading(range(100)):
            trainer.step(train_loader)
            trainer.print_statistics()  # print running loss
            wandb.log({"loss": trainer.loss_history["training_loss"][-1]})
        rvae.save_weights("model")
    except:
        wandb.alert(title="Training failed")
        __import__("pdb").set_trace()


if __name__ == "__main__":
    train()
