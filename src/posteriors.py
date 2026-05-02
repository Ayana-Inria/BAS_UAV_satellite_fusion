"""Posterior generation and model training for RF and U-Net."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PatchDataset
from .models import UNet, UNet2
from .utils import (
    CrossEntropy2d,
    as_hwc,
    count_sliding_window,
    grouper,
    load_array,
    load_pickle,
    save_pickle,
    sliding_window,
    softmax,
    valid_mask,
)


def train_random_forest(
    image: np.ndarray,
    labels: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 0,
    n_jobs: int = -1,
    nodata: float | int | None = 0,
    **kwargs,
) -> RandomForestClassifier:
    image = as_hwc(image)
    labels = np.asarray(labels)
    valid = valid_mask(image, labels, nodata=nodata)
    X = image[valid].reshape(-1, image.shape[-1])
    y = labels[valid].astype(int).ravel()
    if X.size == 0:
        raise ValueError("No valid pixels available to train the Random Forest")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )
    clf.fit(X, y)
    return clf


def random_forest_posterior(
    image: np.ndarray,
    model: RandomForestClassifier | str | Path,
    class_order: Iterable[int] | None = None,
    nodata: float | int | None = 0,
) -> np.ndarray:
    image = as_hwc(image)
    clf = load_pickle(model) if isinstance(model, (str, Path)) else model
    flat = image.reshape(-1, image.shape[-1])
    proba = clf.predict_proba(flat)
    classes = np.asarray(clf.classes_, dtype=int)
    if class_order is not None:
        class_order = np.asarray(list(class_order), dtype=int)
        ordered = np.zeros((flat.shape[0], len(class_order)), dtype=float)
        for src_idx, cls in enumerate(classes):
            dst = np.where(class_order == cls)[0]
            if len(dst):
                ordered[:, dst[0]] = proba[:, src_idx]
        proba = ordered
    posterior = proba.reshape(image.shape[0], image.shape[1], -1)
    if nodata is not None:
        posterior[~valid_mask(image, nodata=nodata)] = 0.0
    return posterior


def build_unet(n_channels: int, n_classes: int, variant: str = "unet", bilinear: bool = True) -> torch.nn.Module:
    key = variant.lower()
    if key in {"unet", "unet1"}:
        return UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    if key in {"unet2", "method2_unet"}:
        return UNet2(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    raise ValueError(f"Unknown U-Net variant: {variant}")


def load_unet_checkpoint(
    checkpoint: str | Path,
    n_channels: int,
    n_classes: int,
    variant: str = "unet",
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = build_unet(n_channels, n_classes, variant=variant).to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
    net.eval()
    return net


def train_unet(
    image: np.ndarray,
    labels: np.ndarray,
    n_channels: int,
    n_classes: int,
    variant: str = "unet",
    epochs: int = 20,
    patch_size: tuple[int, int] = (480, 480),
    samples_per_epoch: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    input_scale: float | None = 255.0,
    device: str | torch.device | None = None,
    save_checkpoint: str | Path | None = None,
) -> torch.nn.Module:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = build_unet(n_channels, n_classes, variant=variant).to(device)
    dataset = PatchDataset(
        image,
        labels,
        patch_size=patch_size,
        samples_per_epoch=samples_per_epoch,
        input_scale=input_scale,
        augmentation=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(1, epochs + 1):
        net.train()
        running = 0.0
        for data, target in tqdm(loader, desc=f"U-Net epoch {epoch}/{epochs}", leave=False):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = net(data)[0]
            loss = CrossEntropy2d(output, target)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        print(f"U-Net epoch {epoch}/{epochs} loss={running / max(1, len(loader)):.6f}")
    net.eval()
    if save_checkpoint:
        path = Path(save_checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), path)
    return net


def _prepare_unet_image(image: np.ndarray, input_scale: float | None) -> np.ndarray:
    image = as_hwc(image).astype("float32")
    if input_scale not in (None, 0, 1):
        image = image / float(input_scale)
    return image


@torch.no_grad()
def unet_multiscale_posteriors(
    image: np.ndarray,
    net: torch.nn.Module,
    window_size: tuple[int, int] = (480, 480),
    stride: int = 240,
    batch_size: int = 1,
    device: str | torch.device | None = None,
    input_scale: float | None = 255.0,
    include_final: bool = True,
    include_activations: bool = True,
) -> list[np.ndarray]:
    image = _prepare_unet_image(image, input_scale)
    device = torch.device(device or next(net.parameters()).device)
    net = net.to(device).eval()
    n_classes = int(getattr(net, "n_classes", 2))
    sums: list[np.ndarray] | None = None
    counts: list[np.ndarray] | None = None
    total = count_sliding_window(image, step=stride, window_size=window_size)
    iterator = grouper(batch_size, sliding_window(image, step=stride, window_size=window_size))
    for coords in tqdm(iterator, total=max(1, total // max(1, batch_size)), desc="U-Net inference", leave=False):
        patches = [np.copy(image[x:x + w, y:y + h]).transpose(2, 0, 1) for x, y, w, h in coords]
        batch = torch.from_numpy(np.asarray(patches)).to(device)
        outputs = net(batch)
        final_logits, activations = outputs if isinstance(outputs, (tuple, list)) else (outputs, [])
        tensors: list[torch.Tensor] = []
        if include_final:
            tensors.append(final_logits)
        if include_activations:
            tensors.extend(list(reversed(activations)))
        if sums is None or counts is None:
            sums, counts = [], []
            for tensor in tensors:
                _, _, ph, pw = tensor.shape
                out_h = max(1, int(round(image.shape[0] * ph / float(window_size[0]))))
                out_w = max(1, int(round(image.shape[1] * pw / float(window_size[1]))))
                sums.append(np.zeros((out_h, out_w, n_classes), dtype="float32"))
                counts.append(np.zeros((out_h, out_w, 1), dtype="float32"))
        for level, tensor in enumerate(tensors):
            logits = tensor.detach().cpu().numpy()
            level_h, level_w = sums[level].shape[:2]
            for out, (x, y, w, h) in zip(logits, coords):
                x0 = int(round(x * level_h / image.shape[0]))
                y0 = int(round(y * level_w / image.shape[1]))
                x1 = min(level_h, x0 + out.shape[1])
                y1 = min(level_w, y0 + out.shape[2])
                patch = out[:, :x1 - x0, :y1 - y0].transpose(1, 2, 0)
                sums[level][x0:x1, y0:y1] += patch
                counts[level][x0:x1, y0:y1] += 1
    if sums is None or counts is None:
        raise ValueError("No U-Net windows were produced")
    result = []
    for logits_sum, count in zip(sums, counts):
        count[count == 0] = 1
        result.append(softmax(logits_sum / count, axis=-1).astype("float32"))
    return result


def unet_posterior(*args, **kwargs) -> np.ndarray:
    return unet_multiscale_posteriors(*args, include_final=True, include_activations=False, **kwargs)[0]
