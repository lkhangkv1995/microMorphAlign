import cv2
import numpy as np
import pytest

from micromorphalign import MicroMorphAlign, structural_feature, plot_match


def tissue(seed=12, size=256):
    rng = np.random.default_rng(seed)
    canvas = np.zeros((size, size), dtype=np.float32)
    for _ in range(170):
        x, y = rng.integers(8, size - 8, size=2)
        cv2.circle(canvas, (int(x), int(y)), int(rng.integers(2, 7)),
                   float(rng.uniform(.3, 1.0)), -1)
    canvas = cv2.GaussianBlur(canvas, (0, 0), .75)
    return np.clip(canvas, 0, 1)


def test_translation_residual_both_directions():
    src = tissue()
    dst = cv2.warpAffine(src, np.float32([[1, 0, 7], [0, 1, -5]]),
                         (256, 256), borderValue=0)
    matcher = MicroMorphAlign(src, dst, matrix12=np.eye(3))
    forward = matcher.map_point((120, 120), context_window=65, search_radius=12,
                                check_cycle=True)
    assert np.allclose(forward["refined"], (127, 115), atol=1.0)
    assert forward["score"] > .60
    assert forward["cycle_error_px"] < 2.0
    backward = matcher.map_point((127, 115), direction="2to1",
                                 context_window=65, search_radius=12)
    assert np.allclose(backward["refined"], (120, 120), atol=1.0)


def test_global_rotation_and_anisotropic_scale():
    src = tissue()
    angle = np.deg2rad(13)
    M = np.array([[1.14*np.cos(angle), -.87*np.sin(angle), 9],
                  [1.14*np.sin(angle), .87*np.cos(angle), -7],
                  [0, 0, 1]], dtype=np.float64)
    dst = cv2.warpPerspective(src, M, (256, 256))
    matcher = MicroMorphAlign(src, dst, matrix12=M)
    result = matcher.map_point((110, 125), context_window=65, search_radius=8)
    assert np.linalg.norm(result["correction"]) < 1.5
    assert result["score"] > .50


def test_multichannel_and_invalid_requests():
    src = tissue()
    multichannel = np.stack([src * (i + 1) for i in range(7)], axis=-1)
    matcher = MicroMorphAlign(multichannel, multichannel, matrix12=np.eye(3))
    result = matcher.map_point((130, 110), search_radius=4)
    assert np.linalg.norm(result["correction"]) < .8
    assert structural_feature(multichannel).shape == src.shape
    with pytest.raises(ValueError, match="odd integer"):
        matcher.map_point((130, 110), context_window=80)


def test_brightfield_rgb_to_single_channel_fluorescence():
    nuclei = tissue(seed=8)
    rgb = np.stack([1 - .65*nuclei, 1 - .50*nuclei,
                    1 - .15*nuclei], axis=-1)
    dapi = cv2.warpAffine(nuclei, np.float32([[1, 0, 4], [0, 1, 6]]),
                          (256, 256), borderValue=0)
    matcher = MicroMorphAlign(rgb, dapi, matrix12=np.eye(3))
    result = matcher.map_point((130, 120), context_window=61, search_radius=10)
    assert np.allclose(result["refined"], (134, 126), atol=1.0)
    assert result["score"] > .40


def test_empty_patch_produces_error():
    blank = np.zeros((128, 128), dtype=np.float32)
    matcher = MicroMorphAlign(blank, blank, matrix12=np.eye(3))
    with pytest.raises(ValueError, match="structurally flat"):
        matcher.map_point((64, 64), context_window=31)


def test_accepts_existing_morphalign_registration_result():
    src = tissue()
    fitted = {"affine": {"matrix_he_to_af": np.eye(3)}, "deformable": None}
    matcher = MicroMorphAlign(src, src.copy(), registration=fitted)
    out = matcher.map_point((120, 118), direction="1to2", search_radius=5)
    assert np.allclose(out["refined"], (120, 118), atol=.5)


def test_plot_shows_source_two_destination_and_registered_patches():
    import matplotlib.pyplot as plt

    image = tissue()
    multi = np.stack([image * (i + 1) for i in range(7)], axis=-1)
    matcher = MicroMorphAlign(multi, image, matrix12=np.eye(3))
    result = matcher.map_point((120, 118), context_window=65, search_radius=4)
    fig = plot_match(matcher, result)
    assert len(fig.axes) == 6
    assert "Image1 at original" in fig.axes[2].get_title()
    assert "Image2 at predicted" in fig.axes[3].get_title()
    assert "Image2 at refined" in fig.axes[4].get_title()
    assert "Refined Image2 → Image1 grid" in fig.axes[5].get_title()
    plt.close(fig)


def test_final_registered_patch_centers_on_refined_destination_point():
    from micromorphalign.plotting import _raw_patch

    src = tissue()
    dst = cv2.warpAffine(src, np.float32([[1, 0, 7], [0, 1, -5]]),
                         (256, 256), borderValue=0)
    matcher = MicroMorphAlign(src, dst, matrix12=np.eye(3))
    r = matcher.map_point((120, 120), context_window=65, search_radius=12)
    local = _raw_patch(dst, r["refined"], 65)
    registered = _raw_patch(dst, r["source_point"], 65,
                            output_to_image=matcher.matrix12,
                            correction=r["correction"])
    assert np.isclose(registered[32, 32], local[32, 32], atol=1e-4)
