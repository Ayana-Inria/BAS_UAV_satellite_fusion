"""Shared utilities: config, IO, image helpers, sliding windows, and metrics."""

from __future__ import annotations

import itertools
import pickle
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import confusion_matrix

from .config import INVERT_PALETTE, LABELS, PALETTE

palette = PALETTE
invert_palette = INVERT_PALETTE


def read_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def require(cfg: dict, key: str) -> Any:
    current: Any = cfg
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Missing required config key: {key}")
        current = current[part]
    return current


def load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        if len(data.files) != 1:
            raise ValueError(f"{path} contains multiple arrays; use .npy or save a single-array .npz")
        return data[data.files[0]]
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return np.asarray(Image.open(path))
    raise ValueError(f"Unsupported array format: {path}")


def save_array(array: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, array)
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        Image.fromarray(np.asarray(array)).save(path)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def save_labels(labels: np.ndarray, path: str | Path, color: bool = False) -> None:
    arr = convert_to_color(labels) if color else labels.astype(np.uint8)
    save_array(arr, path)


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def as_hwc(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D image, got shape {arr.shape}")
    if arr.shape[0] <= 32 and arr.shape[-1] > 32:
        return np.moveaxis(arr, 0, -1)
    return arr


def valid_mask(image: np.ndarray, labels: np.ndarray | None = None, nodata: float | int | None = 0) -> np.ndarray:
    img = as_hwc(image)
    valid = np.ones(img.shape[:2], dtype=bool)
    if nodata is not None:
        valid &= np.any(img != nodata, axis=-1)
    if labels is not None:
        lab = np.asarray(labels)
        if lab.shape != valid.shape:
            raise ValueError(f"labels shape {lab.shape} does not match image shape {valid.shape}")
        valid &= lab >= 0
    return valid


def convert_to_color(arr_2d: np.ndarray, palette: dict[int, tuple[int, int, int]] = palette) -> np.ndarray:
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, color in palette.items():
        arr_3d[arr_2d == c] = color
    return arr_3d


def convert_from_color(arr_3d: np.ndarray, palette: dict[tuple[int, int, int], int] = invert_palette) -> np.ndarray:
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for color, c in palette.items():
        arr_2d[np.all(arr_3d == np.array(color).reshape(1, 1, 3), axis=2)] = c
    return arr_2d


def get_random_pos1(img: np.ndarray, window_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = window_shape
    W, H = img.shape[-2:]
    if W < w or H < h:
        raise ValueError(f"Image shape {img.shape} is smaller than requested window {window_shape}")
    x1 = random.randint(0, max(0, W - w))
    y1 = random.randint(0, max(0, H - h))
    return x1, x1 + w, y1, y1 + h


def sliding_window(top: np.ndarray, step: int = 10, window_size: tuple[int, int] = (20, 20)):
    H, W = top.shape[:2]
    wh, ww = window_size
    if H < wh or W < ww:
        raise ValueError(f"Image shape {(H, W)} is smaller than window_size {window_size}")
    seen = set()
    for x in range(0, H, step):
        x = min(x, H - wh)
        for y in range(0, W, step):
            y = min(y, W - ww)
            if (x, y) not in seen:
                seen.add((x, y))
                yield x, y, wh, ww


def count_sliding_window(top: np.ndarray, step: int = 10, window_size: tuple[int, int] = (20, 20)) -> int:
    return sum(1 for _ in sliding_window(top, step=step, window_size=window_size))


def grouper(n: int, iterable: Iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def softmax(X: np.ndarray, theta: float = 1.0, axis: int | None = None) -> np.ndarray:
    y = np.atleast_2d(X).astype(float) * float(theta)
    if axis is None:
        axis = next(j for j, size in enumerate(y.shape) if size > 1)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)
    p = y / np.expand_dims(np.sum(y, axis=axis), axis)
    return p.flatten() if len(X.shape) == 1 else p


def CrossEntropy2d(input: torch.Tensor, target: torch.Tensor, weight=None, reduction: str = "mean"):
    if input.dim() == 2:
        return F.cross_entropy(input, target, weight=weight, reduction=reduction, ignore_index=-1)
    if input.dim() == 4:
        output = input.permute(0, 2, 3, 1).contiguous().view(-1, input.size(1))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight=weight, reduction=reduction, ignore_index=-1)
    raise ValueError(f"Expected 2 or 4 dimensions, got {input.dim()}")


def accuracy(input: np.ndarray, target: np.ndarray) -> float:
    return 100.0 * float(np.count_nonzero(input == target)) / target.size


def metrics(predictions: np.ndarray, gts: np.ndarray, label_values=LABELS):
    cm = confusion_matrix(gts, predictions)
    total = np.sum(cm)
    overall = 100.0 * np.trace(cm) / float(total) if total else 0.0
    print("Confusion matrix:")
    print(cm)
    print(f"Total accuracy: {overall:.4f}%")
    return overall
