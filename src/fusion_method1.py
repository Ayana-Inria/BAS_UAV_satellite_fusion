"""Method 1: pixelwise probabilistic Sentinel-2/drone fusion."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .config import BATCH_SIZE, N_CLASSES, WINDOW_SIZE
from .utils import count_sliding_window, grouper, sliding_window


def compute_prior(drone_img: np.ndarray, labels: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    valid = np.any(drone_img, axis=-1) if drone_img.ndim == 3 else np.ones_like(labels, dtype=bool)
    selected = labels[valid].astype(int)
    counts = np.bincount(selected.ravel(), minlength=n_classes)[:n_classes].astype(float)
    total = counts.sum()
    if total == 0:
        raise ValueError("Cannot compute prior: no valid pixels found")
    return counts / total


def conditional_probability_matrix(epsilon_1: float, epsilon_2: float, alpha: float) -> np.ndarray:
    return np.array(
        [[1 - epsilon_1, epsilon_2, alpha],
         [epsilon_1, 1 - epsilon_2, 1 - alpha]],
        dtype=float,
    )


def pixelwise_prob_fus_EM(
    posterior_sentinel: np.ndarray,
    drone_posteriors: np.ndarray,
    drone_prior: np.ndarray,
    cond_prob_mat: np.ndarray,
    stride: int = WINDOW_SIZE[0],
    batch_size: int = BATCH_SIZE,
    window_size: tuple[int, int] = WINDOW_SIZE,
    sentinel_scale: int = 480,
) -> list[np.ndarray]:
    posterior_sentinel = np.asarray(posterior_sentinel, dtype=float)
    drone_posteriors = np.asarray(drone_posteriors, dtype=float)
    drone_prior = np.asarray(drone_prior, dtype=float)
    cond_prob_mat = np.asarray(cond_prob_mat, dtype=float)
    if drone_posteriors.ndim != 3 or posterior_sentinel.ndim != 3:
        raise ValueError("posterior_sentinel and drone_posteriors must be H x W x C arrays")
    n_classes = drone_posteriors.shape[-1]
    if len(drone_prior) != n_classes:
        raise ValueError(f"drone_prior length {len(drone_prior)} does not match classes {n_classes}")
    if cond_prob_mat.shape[0] != n_classes:
        raise ValueError("cond_prob_mat rows must match drone classes")
    if cond_prob_mat.shape[1] != posterior_sentinel.shape[-1]:
        raise ValueError("cond_prob_mat columns must match Sentinel posterior classes")
    pred = np.zeros(drone_posteriors.shape[:2] + (n_classes,), dtype=float)
    counts = np.zeros(drone_posteriors.shape[:2] + (1,), dtype=float)
    total = count_sliding_window(drone_posteriors[:, :, 0], step=stride, window_size=window_size)
    iterator = grouper(batch_size, sliding_window(drone_posteriors[:, :, 0], step=stride, window_size=window_size))
    for coords in tqdm(iterator, total=max(1, total // max(1, batch_size)), desc="Method 1 fusion", leave=False):
        for x, y, w, h in coords:
            sx, sy = int(x / sentinel_scale), int(y / sentinel_scale)
            sw, sh = max(1, int(w / sentinel_scale)), max(1, int(h / sentinel_scale))
            sentinel_patch = posterior_sentinel[sx:sx + sw, sy:sy + sh]
            if sentinel_patch.size == 0:
                continue
            sentinel_vec = sentinel_patch.reshape(-1, sentinel_patch.shape[-1]).mean(axis=0)
            discr = cond_prob_mat @ sentinel_vec
            drone_patch = drone_posteriors[x:x + w, y:y + h]
            log_res = (
                np.log(drone_patch + 1e-12)
                + np.log(discr + 1e-12).reshape(1, 1, -1)
                - np.log(drone_prior + 1e-12).reshape(1, 1, -1)
            )
            pred[x:x + w, y:y + h] += log_res
            counts[x:x + w, y:y + h] += 1
    counts[counts == 0] = 1
    return [np.argmax(pred / counts, axis=-1).astype(np.uint8)]
