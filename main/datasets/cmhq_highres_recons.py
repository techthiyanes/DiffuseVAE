import numpy as np
import os
import torch

from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image


class CMHQHighResReconsDataset(Dataset):
    def __init__(
        self,
        cmhq_root,
        recons_root,
        norm=False,
        transform=None,
        **kwargs,
    ):
        if not os.path.isdir(cmhq_root):
            raise ValueError(f"The specified root: {cmhq_root} does not exist")
        self.cmhq_root = cmhq_root
        self.recons_root = recons_root
        self.transform = transform
        self.norm = norm

        self.images = []

        img_path = os.path.join(self.cmhq_root, "CelebA-HQ-img")
        for img in tqdm(os.listdir(img_path)):
            self.images.append(os.path.join(img_path, img))

        self.recons = np.load(
            os.path.join(self.recons_root, "annotated_recons_celebamaskhq128.npy")
        )

        assert self.images.shape[0] == self.recons.shape[0]

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)
        recons = torch.from_numpy(self.recons[idx])

        if self.transform is not None:
            img = self.transform(img)
            recons = self.transform(recons)

        # Normalize between (-1, 1) (Assuming between [0, 1])
        if self.norm:
            img = (np.asarray(img).astype(np.float) / 127.5) - 1.0
            recons = 2 * recons - 1
        else:
            img = np.asarray(img).astype(np.float) / 255.0
        return recons, torch.from_numpy(img).permute(2, 0, 1).float()

    def __len__(self):
        return len(self.images)
