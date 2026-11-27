"""Channel-agnostic, lightweight cross-modality structural features."""

from __future__ import annotations

import cv2
import numpy as np


def _scale(image: np.ndarray) -> np.ndarray:
    """Robust float32 scaling without assuming 8-bit input."""
    a = np.nan_to_num(np.asarray(image, dtype=np.float32))
    lo, hi = np.percentile(a, [1, 99.5])
    return np.clip((a - lo) / max(float(hi - lo), 1e-6), 0, 1)


def structural_feature(image: np.ndarray, mode: str = "auto") -> np.ndarray:
    """Derive comparable 2D maps from HxW or HxWxC tissue images.

    ``auto`` emphasizes locally bright, nucleus-like structures (inverting
    brightfield-like RGB images), plus intensity-invariant edges. ``edges``
    matches gradients alone, useful when staining phenotypes differ strongly.
    This is a heuristic, not a guarantee of cross-modality correspondence.
    """
    raw = np.asarray(image)
    if raw.ndim == 2:
        signal = _scale(raw)
    elif raw.ndim == 3 and raw.shape[2] >= 1:
        channels = [_scale(raw[..., i]) for i in range(raw.shape[2])]
        brightfield = raw.shape[2] == 3 and float(
            np.median(np.mean(np.stack(channels, axis=-1), axis=-1))
        ) > .55
        if brightfield:
            signal = 1 - (.299 * channels[0] + .587 * channels[1] + .114 * channels[2])
        else:
            # Prefer the channel with the strongest local morphology, not
            # necessarily the one with the largest overall fluorescence.
            contrast = [float(np.std(c - cv2.GaussianBlur(c, (0, 0), 3)))
                        for c in channels]
            signal = channels[int(np.argmax(contrast))]
    else:
        raise ValueError("Images must be HxW or channel-last HxWxC arrays")

    if mode not in {"auto", "edges"}:
        raise ValueError("feature mode must be 'auto' or 'edges'")

    fine = cv2.GaussianBlur(signal.astype(np.float32), (0, 0), 1.25)
    gx = cv2.Sobel(fine, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(fine, cv2.CV_32F, 0, 1, ksize=3)
    edges = _scale(cv2.magnitude(gx, gy))
    if mode == "edges":
        return edges.astype(np.float32)

    background = cv2.GaussianBlur(fine, (0, 0), 7.0)
    nuclei = _scale(np.maximum(fine - background, 0))
    return (.75 * nuclei + .25 * edges).astype(np.float32)
