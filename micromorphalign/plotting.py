"""Publication-friendly visualization of a MicroMorphAlign point match."""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _show_image(ax, image):
    raw = np.asarray(image)
    if raw.ndim == 3:
        planes = []
        for i in range(min(3, raw.shape[-1])):
            plane = raw[..., i].astype(np.float32)
            lo, hi = np.percentile(plane, [1, 99.5])
            planes.append(np.clip((plane - lo) / max(hi - lo, 1e-6), 0, 1))
        if len(planes) == 1:
            ax.imshow(planes[0], cmap="gray")
        else:
            while len(planes) < 3:
                planes.append(np.zeros_like(planes[0]))
            ax.imshow(np.stack(planes, axis=-1))
    else:
        ax.imshow(raw, cmap="gray")


def _raw_patch(image, center, size, *, output_to_image=None, correction=(0, 0)):
    """Sample a patch directly or through an output-to-image affine mapping.

    Sampling only the output patch avoids creating enormous image copies and
    works for arbitrarily many channel-last channels.
    """
    half = size // 2
    offset = np.arange(size, dtype=np.float32) - half
    xx, yy = np.meshgrid(offset + float(center[0]),
                         offset + float(center[1]))
    if output_to_image is not None:
        M = np.asarray(output_to_image, dtype=np.float64)
        x0, y0 = xx.copy(), yy.copy()
        xx = M[0, 0] * x0 + M[0, 1] * y0 + M[0, 2] + float(correction[0])
        yy = M[1, 0] * x0 + M[1, 1] * y0 + M[1, 2] + float(correction[1])
    xx, yy = xx.astype(np.float32), yy.astype(np.float32)
    raw = np.asarray(image)
    if raw.ndim == 2:
        return cv2.remap(raw, xx, yy, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.stack([
        cv2.remap(raw[..., channel], xx, yy, interpolation=cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        for channel in range(raw.shape[-1])
    ], axis=-1)


def plot_match(matcher, result, *, figsize=(24, 4.4)):
    """Display source → destination prediction → refinement → registered patch.

    The two middle patches are raw crops from the destination image at its
    native scale. The final patch samples the *refined destination tissue*
    after affine resampling into the source image's pixel grid, including the
    point-level translation correction. It is thus at the same scale as the
    original source patch, and contains no pixels from the source image.
    """
    direction = result["direction"]
    source = matcher.image1 if direction == "1to2" else matcher.image2
    dest = matcher.image2 if direction == "1to2" else matcher.image1
    source_label = "Image1" if direction == "1to2" else "Image2"
    dest_label = "Image2" if direction == "1to2" else "Image1"
    source_to_dest = matcher.matrix12 if direction == "1to2" else matcher.matrix21
    size = int(result["template"].shape[0])

    native_patch = _raw_patch(source, result["source_point"], size)
    predicted_patch = _raw_patch(dest, result["predicted"], size)
    final_patch = _raw_patch(dest, result["refined"], size)
    registered_patch = _raw_patch(dest, result["source_point"], size,
                                  output_to_image=source_to_dest,
                                  correction=result["correction"])

    fig, axes = plt.subplots(1, 6, figsize=figsize)
    _show_image(axes[0], source)
    axes[0].plot(*result["source_point"], "r+", markersize=12, markeredgewidth=2)
    axes[0].set_title("Source image · original point")

    _show_image(axes[1], dest)
    axes[1].plot(*result["predicted"], marker="x", color="orange", markersize=10,
                 markeredgewidth=2, linestyle="None", label="Global")
    axes[1].plot(*result["refined"], marker="+", color="lime", markersize=12,
                 markeredgewidth=2, linestyle="None", label="Refined")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Destination · global vs refined")

    for ax, patch in zip(axes[2:],
                         (native_patch, predicted_patch, final_patch, registered_patch)):
        _show_image(ax, patch)
    axes[2].set_title(f"1 · {source_label} at original (x, y)")
    axes[3].set_title(f"2 · {dest_label} at predicted (x_w, y_w)")
    axes[4].set_title(f"3 · {dest_label} at refined (x_r, y_r)")
    axes[5].set_title(f"4 · Refined {dest_label} → {source_label} grid")

    half = size // 2
    axes[2].plot(half, half, "+", color="red", markersize=10, markeredgewidth=1.5)
    axes[3].plot(half, half, "x", color="orange", markersize=10, markeredgewidth=1.5)
    axes[4].plot(half, half, "+", color="lime", markersize=10, markeredgewidth=1.5)
    axes[5].plot(half, half, "+", color="lime", markersize=10, markeredgewidth=1.5)

    # Show where the global prediction would fall relative to the final match.
    axes[4].plot(half + result["predicted"][0] - result["refined"][0],
                 half + result["predicted"][1] - result["refined"][1],
                 marker="x", color="orange", markersize=9,
                 markeredgewidth=1.6, linestyle="None")

    for a in axes:
        a.axis("off")
    fig.suptitle(f"Local refinement · NCC {result['score']:.3f} · "
                 f"correction ({result['correction'][0]:+.2f}, "
                 f"{result['correction'][1]:+.2f}) px", fontsize=12)
    fig.tight_layout()
    return fig
