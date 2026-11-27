"""Bidirectional local correspondence refinement after MorphAlign registration.

Coordinate convention: (x, y) with zero-based pixel-center coordinates.
M12 maps original image1 points into original image2 pixel coordinates.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

from .features import structural_feature


def _matrix3(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0, 0, 1]])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 3x3 or 2x3 affine matrix")
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Affine matrix is singular")
    return matrix


def _transform(matrix, xy):
    pt = np.asarray(xy, dtype=np.float64)
    if pt.shape != (2,) or not np.isfinite(pt).all():
        raise ValueError("point must be a finite (x, y) pair")
    out = matrix @ np.r_[pt, 1.0]
    if abs(out[2]) < 1e-12:
        raise ValueError("Invalid homogeneous coordinate")
    return out[:2] / out[2]


def _parabolic(a, b, c):
    den = float(a - 2 * b + c)
    return 0.0 if abs(den) < 1e-8 else float(np.clip(.5 * (a - c) / den, -.5, .5))


class MicroMorphAlign:
    """Refine landmark coordinates in either direction with a cached feature map.

    Supply ``registration`` (from ``morphalign.register``), ``matrix12`` or
    neither. With neither, the underlying global affine registration runs once.
    A pre-existing deformable registration is currently reduced to its *affine*
    component; deformation fields are not yet used to map point coordinates.
    """

    def __init__(self, image1, image2, *, registration=None, matrix12=None,
                 feature_mode="auto", feature1=None, feature2=None,
                 register_kwargs=None):
        self.image1 = np.asarray(image1)
        self.image2 = np.asarray(image2)
        for name, image in (("image1", self.image1), ("image2", self.image2)):
            if image.ndim not in (2, 3):
                raise ValueError(f"{name} must be HxW or HxWxC")

        if matrix12 is not None and registration is not None:
            raise ValueError("Pass matrix12 OR registration, not both")
        if registration is None and matrix12 is None:
            from morphalign import register
            kwargs = dict(register_kwargs or {})
            if kwargs.get("deformable", False):
                raise ValueError("Point coordinate mapping currently requires deformable=False")
            kwargs["deformable"] = False
            registration = register(self.image1, self.image2, **kwargs)

        if registration is not None:
            if registration.get("deformable") is not None:
                warnings.warn("Using the affine component only; deformable coordinate mapping is not supported", stacklevel=2)
            fitted = registration.get("affine", registration)
            matrix12 = fitted["matrix_he_to_af"]

        self.registration = registration
        self.matrix12 = _matrix3(matrix12)
        self.matrix21 = np.linalg.inv(self.matrix12)

        self.feature1 = (structural_feature(self.image1, feature_mode)
                         if feature1 is None else np.asarray(feature1, dtype=np.float32))
        self.feature2 = (structural_feature(self.image2, feature_mode)
                         if feature2 is None else np.asarray(feature2, dtype=np.float32))
        if self.feature1.shape != self.image1.shape[:2] or self.feature2.shape != self.image2.shape[:2]:
            raise ValueError("Custom feature arrays must match their image's spatial dimensions")
        self._warped = {}

    def _direction_data(self, direction):
        if direction == "1to2":
            return self.feature1, self.feature2, self.matrix12
        if direction == "2to1":
            return self.feature2, self.feature1, self.matrix21
        raise ValueError("direction must be '1to2' or '2to1'")

    def _aligned_source(self, direction):
        if direction not in self._warped:
            source, destination, matrix = self._direction_data(direction)
            h, w = destination.shape
            warped = cv2.warpPerspective(source, matrix, (w, h),
                                         flags=cv2.INTER_LINEAR, borderValue=0)
            valid = cv2.warpPerspective(np.ones(source.shape, np.uint8), matrix,
                                        (w, h), flags=cv2.INTER_NEAREST,
                                        borderValue=0).astype(np.float32)
            self._warped[direction] = warped, valid
        return self._warped[direction]

    def map_point(self, point, *, direction="1to2", context_window=81,
                  search_radius=20, min_coverage=.90, check_cycle=False):
        """Return globally predicted and locally refined (x,y) coordinates.

        ``context_window`` and ``search_radius`` are expressed in *destination*
        pixels. All source patches are resampled onto the destination grid,
        accounting for global rotation, anisotropic scale and translation.
        Translation-only residuals are searched; do not interpret NCC as truth.
        """
        size = int(context_window)
        radius = int(search_radius)
        if size < 3 or size % 2 == 0:
            raise ValueError("context_window must be an odd integer >=3")
        if radius < 0 or not 0 <= min_coverage <= 1:
            raise ValueError("Invalid search_radius or min_coverage")
        _, destination, matrix = self._direction_data(direction)
        predicted = _transform(matrix, point)
        x, y = map(float, predicted)
        height, width = destination.shape
        half = size // 2
        if not (half <= x <= width - half - 1 and half <= y <= height - half - 1):
            raise ValueError("Predicted point too close to destination border for context_window")

        warped, valid = self._aligned_source(direction)
        template = cv2.getRectSubPix(warped, (size, size), (x, y))
        coverage = float(cv2.getRectSubPix(valid, (size, size), (x, y)).mean())
        if coverage < min_coverage:
            raise ValueError(f"Source patch coverage {coverage:.1%} below threshold {min_coverage:.1%}")
        if float(np.std(template)) < 1e-5:
            raise ValueError("Source patch is structurally flat; choose another point/window")

        xmin = max(half, int(np.ceil(x - radius)))
        xmax = min(width - half - 1, int(np.floor(x + radius)))
        ymin = max(half, int(np.ceil(y - radius)))
        ymax = min(height - half - 1, int(np.floor(y + radius)))
        if xmin > xmax or ymin > ymax:
            raise ValueError("No complete destination patches within search radius")

        region = destination[ymin - half:ymax + half + 1,
                             xmin - half:xmax + half + 1]
        scores = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite NCC scores (insufficient patch contrast)")
        py, px = np.unravel_index(int(np.argmax(scores)), scores.shape)
        dx = (_parabolic(scores[py, px - 1], scores[py, px], scores[py, px + 1])
              if 0 < px < scores.shape[1] - 1 else 0.0)
        dy = (_parabolic(scores[py - 1, px], scores[py, px], scores[py + 1, px])
              if 0 < py < scores.shape[0] - 1 else 0.0)
        refined = np.array([xmin + px + dx, ymin + py + dy], dtype=float)

        alternatives = scores.copy()
        alternatives[max(0, py - 3):min(scores.shape[0], py + 4),
                     max(0, px - 3):min(scores.shape[1], px + 4)] = -np.inf
        remaining = alternatives[np.isfinite(alternatives)]
        peak_margin = (float(scores[py, px] - remaining.max())
                       if remaining.size else float("nan"))

        result = {
            "direction": direction,
            "source_point": tuple(map(float, point)),
            "predicted": tuple(map(float, predicted)),
            "refined": tuple(map(float, refined)),
            "correction": tuple(map(float, refined - predicted)),
            "score": float(scores[py, px]),
            "peak_margin": peak_margin,
            "coverage": coverage,
            "template": template,
            "matched_patch": cv2.getRectSubPix(destination, (size, size),
                                                tuple(map(float, refined))),
            "score_map": scores,
            "search_origin": (xmin, ymin),
        }
        if check_cycle:
            reverse = "2to1" if direction == "1to2" else "1to2"
            back = self.map_point(refined, direction=reverse,
                                  context_window=size, search_radius=radius,
                                  min_coverage=min_coverage, check_cycle=False)
            result["cycle_error_px"] = float(np.linalg.norm(
                np.asarray(back["refined"]) - np.asarray(point)))
            result["reverse_score"] = back["score"]
        return result

    def map_points(self, points, **kwargs):
        """Map an iterable of points, reusing cached source feature warps."""
        return [self.map_point(p, **kwargs) for p in points]


def refine_point(image1, image2, point, *, matrix12=None, registration=None,
                 direction="1to2", context_window=81, search_radius=20,
                 feature_mode="auto", check_cycle=False):
    """Convenience wrapper; prefer MicroMorphAlign for multiple points."""
    matcher = MicroMorphAlign(image1, image2, registration=registration,
                              matrix12=matrix12, feature_mode=feature_mode)
    return matcher.map_point(point, direction=direction,
                             context_window=context_window,
                             search_radius=search_radius, check_cycle=check_cycle)
