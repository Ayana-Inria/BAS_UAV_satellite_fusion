"""Run Method 1: Sentinel-2 RF posterior + drone UNet posterior + pixelwise fusion."""

from __future__ import annotations

import argparse
import numpy as np

from src.fusion_method1 import conditional_probability_matrix, pixelwise_prob_fus_EM
from src.pipeline import drone_posterior_from_unet, sentinel2_posterior_from_rf
from src.utils import read_config, require, save_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/method1.example.yaml")
    args = parser.parse_args()

    cfg = read_config(args.config)
    posterior_sentinel = sentinel2_posterior_from_rf(require(cfg, "inputs.sentinel2"))
    drone_posteriors = drone_posterior_from_unet(require(cfg, "inputs.drone"))

    cp = require(cfg, "conditional_probability")
    cond = conditional_probability_matrix(
        epsilon_1=float(cp["epsilon_1"]),
        epsilon_2=float(cp["epsilon_2"]),
        alpha=float(cp["alpha"]),
    )
    labels = pixelwise_prob_fus_EM(
        posterior_sentinel=posterior_sentinel,
        drone_posteriors=drone_posteriors,
        drone_prior=np.asarray(require(cfg, "prior_drone"), dtype=float),
        cond_prob_mat=cond,
        stride=int(cfg.get("stride", 240)),
        batch_size=int(cfg.get("batch_size", 1)),
        window_size=tuple(cfg.get("window_size", [480, 480])),
        sentinel_scale=int(cfg.get("sentinel_scale", 480)),
    )[0]
    save_labels(labels, require(cfg, "output.labels"), color=bool(cfg.get("output", {}).get("color", False)))
    print(f"Saved Method 1 labels to {require(cfg, 'output.labels')}")


if __name__ == "__main__":
    main()
