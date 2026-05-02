"""Run Method 2: Sentinel-2 RF posterior + UNet2 multiscale drone posteriors + MRF fusion."""

from __future__ import annotations

import argparse
import numpy as np

from src.mrf import bottom_up, computePrior, computeTransProb
from src.pipeline import drone_multiscale_posteriors_from_unet2, sentinel2_posterior_from_rf
from src.utils import load_array, read_config, require, save_labels


def _to_mrf_layout(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] <= 32:
        arr = np.moveaxis(arr, -1, 0)[..., None]
    elif arr.ndim == 3:
        arr = arr[..., None]
    if arr.ndim != 4:
        raise ValueError("Each posterior must be H x W x C, C x H x W, or C x H x W x 1")
    return arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/method2.example.yaml")
    args = parser.parse_args()

    cfg = read_config(args.config)
    sentinel = sentinel2_posterior_from_rf(require(cfg, "inputs.sentinel2"))
    drone_2cm, drone_4cm, drone_8cm = drone_multiscale_posteriors_from_unet2(require(cfg, "inputs.drone"))

    rf_list = [_to_mrf_layout(arr) for arr in [sentinel, drone_8cm, drone_4cm, drone_2cm]]
    R = len(rf_list)
    C, H, W, _ = rf_list[0].shape
    theta = float(cfg.get("theta", 0.9))
    resolution_tran_prob = [computeTransProb(C, theta) for _ in range(R)]

    map_paths = cfg.get("inputs", {}).get("maps_for_prior")
    if map_paths:
        maps = [load_array(p).astype(int) for p in map_paths]
        prior = computePrior(maps, R, C, resolution_tran_prob)
    else:
        prior = [np.asarray(cfg.get("prior", [1 / C] * C), dtype=np.float32) for _ in range(R)]

    partial_post = bottom_up(rf_list, resolution_tran_prob, prior, R, C, H, W, cfg.get("method", "MRF"))
    labels = np.argmax(partial_post[0], axis=0).astype(np.uint8)
    save_labels(labels, require(cfg, "output.labels"), color=bool(cfg.get("output", {}).get("color", False)))
    print(f"Saved Method 2 labels to {require(cfg, 'output.labels')}")


if __name__ == "__main__":
    main()
