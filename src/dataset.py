"""PyTorch datasets for U-Net training from image/label arrays."""

from __future__ import annotations

import random

import numpy as np
import torch

from .utils import as_hwc


class PatchDataset(torch.utils.data.Dataset):
    """Random patch dataset for semantic segmentation.

    Parameters
    ----------
    images:
        One H x W x C image or a list of images.
    labels:
        One H x W label map or a list of label maps.
    patch_size:
        Patch height and width.
    samples_per_epoch:
        Number of random patches returned per epoch.
    input_scale:
        Divide image values by this number. Use ``1`` for already-normalized data.
    augmentation:
        Apply random flips.
    """

    def __init__(
        self,
        images,
        labels,
        patch_size: tuple[int, int] = (480, 480),
        samples_per_epoch: int = 10000,
        input_scale: float | None = 255.0,
        augmentation: bool = True,
    ):
        self.images = [as_hwc(img).astype("float32") for img in (images if isinstance(images, list) else [images])]
        self.labels = [np.asarray(lbl, dtype="int64") for lbl in (labels if isinstance(labels, list) else [labels])]
        self.patch_size = tuple(patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.input_scale = input_scale
        self.augmentation = augmentation
        if len(self.images) != len(self.labels):
            raise ValueError("images and labels must have the same length")
        for img, lab in zip(self.images, self.labels):
            if img.shape[:2] != lab.shape:
                raise ValueError(f"label shape {lab.shape} does not match image shape {img.shape[:2]}")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        i = random.randrange(len(self.images))
        image = self.images[i]
        label = self.labels[i]
        ph, pw = self.patch_size
        H, W = label.shape
        if H < ph or W < pw:
            raise ValueError(f"Image {(H, W)} is smaller than patch_size {self.patch_size}")
        x = random.randint(0, H - ph)
        y = random.randint(0, W - pw)
        patch = image[x:x + ph, y:y + pw]
        target = label[x:x + ph, y:y + pw]
        if self.augmentation:
            if random.random() < 0.5:
                patch = patch[::-1, :, :]
                target = target[::-1, :]
            if random.random() < 0.5:
                patch = patch[:, ::-1, :]
                target = target[:, ::-1]
        if self.input_scale not in (None, 0, 1):
            patch = patch / float(self.input_scale)
        patch = np.copy(patch.transpose(2, 0, 1))
        target = np.copy(target)
        return torch.from_numpy(patch), torch.from_numpy(target)
