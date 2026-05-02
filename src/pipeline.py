"""Pipeline functions for Method 1 and Method 2."""

from __future__ import annotations

import numpy as np

from .posteriors import (
    load_unet_checkpoint,
    random_forest_posterior,
    train_random_forest,
    train_unet,
    unet_multiscale_posteriors,
    unet_posterior,
)
from .utils import load_array, load_pickle, require, save_array, save_pickle


def sentinel2_posterior_from_rf(cfg: dict) -> np.ndarray:
    """Create Sentinel-2 posterior with RF; train if no pre-trained RF is supplied."""
    image = load_array(require(cfg, "image"))
    class_order = cfg.get("class_order")
    if "model" in cfg:
        clf = load_pickle(cfg["model"])
    else:
        labels = load_array(require(cfg, "train_labels"))
        clf = train_random_forest(
            image=image,
            labels=labels,
            n_estimators=int(cfg.get("n_estimators", 100)),
            random_state=int(cfg.get("random_state", 0)),
            n_jobs=int(cfg.get("n_jobs", -1)),
            nodata=cfg.get("nodata", 0),
        )
        if "save_model" in cfg:
            save_pickle(clf, cfg["save_model"])
    posterior = random_forest_posterior(
        image=image,
        model=clf,
        class_order=class_order,
        nodata=cfg.get("nodata", 0),
    )
    if "save_posterior" in cfg:
        save_array(posterior, cfg["save_posterior"])
    return posterior


def _unet_from_cfg(cfg: dict):
    image = load_array(require(cfg, "image"))
    n_channels = int(cfg.get("n_channels", image.shape[-1] if image.ndim == 3 else 1))
    n_classes = int(cfg.get("n_classes", 2))
    variant = cfg.get("variant", "unet")
    if "checkpoint" in cfg:
        net = load_unet_checkpoint(
            checkpoint=cfg["checkpoint"],
            n_channels=n_channels,
            n_classes=n_classes,
            variant=variant,
            device=cfg.get("device"),
        )
    else:
        labels = load_array(require(cfg, "train_labels"))
        net = train_unet(
            image=image,
            labels=labels,
            n_channels=n_channels,
            n_classes=n_classes,
            variant=variant,
            epochs=int(cfg.get("epochs", 20)),
            patch_size=tuple(cfg.get("patch_size", cfg.get("window_size", [480, 480]))),
            samples_per_epoch=int(cfg.get("samples_per_epoch", 1000)),
            batch_size=int(cfg.get("train_batch_size", cfg.get("batch_size", 4))),
            learning_rate=float(cfg.get("learning_rate", 1e-3)),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
            input_scale=cfg.get("input_scale", 255.0),
            device=cfg.get("device"),
            save_checkpoint=cfg.get("save_checkpoint"),
        )
    return image, net


def drone_posterior_from_unet(cfg: dict) -> np.ndarray:
    """Create 2 cm drone posterior with UNet; train if no checkpoint is supplied."""
    image, net = _unet_from_cfg(cfg)
    posterior = unet_posterior(
        image=image,
        net=net,
        window_size=tuple(cfg.get("window_size", [480, 480])),
        stride=int(cfg.get("stride", 240)),
        batch_size=int(cfg.get("infer_batch_size", cfg.get("batch_size", 1))),
        device=cfg.get("device"),
        input_scale=cfg.get("input_scale", 255.0),
    )
    if "save_posterior" in cfg:
        save_array(posterior, cfg["save_posterior"])
    return posterior


def drone_multiscale_posteriors_from_unet2(cfg: dict) -> list[np.ndarray]:
    """Create 2 cm, 4 cm, and 8 cm drone posteriors with UNet2.

    The 4 cm and 8 cm posteriors come from UNet2 intermediate activation logits,
    softmaxed after tiled inference.
    """
    cfg = dict(cfg)
    cfg["variant"] = "unet2"
    image, net = _unet_from_cfg(cfg)
    posteriors = unet_multiscale_posteriors(
        image=image,
        net=net,
        window_size=tuple(cfg.get("window_size", [480, 480])),
        stride=int(cfg.get("stride", 240)),
        batch_size=int(cfg.get("infer_batch_size", cfg.get("batch_size", 1))),
        device=cfg.get("device"),
        input_scale=cfg.get("input_scale", 255.0),
        include_final=True,
        include_activations=True,
    )
    if len(posteriors) < 3:
        raise ValueError("UNet2 did not return enough outputs for 2 cm, 4 cm, and 8 cm posteriors")
    outputs = cfg.get("save_posteriors", {})
    for key, posterior in zip(["drone_2cm", "drone_4cm", "drone_8cm"], posteriors[:3]):
        if key in outputs:
            save_array(posterior, outputs[key])
    return posteriors[:3]


def resolve_posterior(block: dict | str) -> np.ndarray:
    if isinstance(block, str):
        return load_array(block)
    if "path" in block:
        return load_array(block["path"])
    source = require(block, "source").lower()
    if source == "rf":
        return sentinel2_posterior_from_rf(block)
    if source == "unet":
        return drone_posterior_from_unet(block)
    raise ValueError(f"Unsupported posterior source: {source}")
