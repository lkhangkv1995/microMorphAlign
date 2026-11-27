"""MicroMorphAlign: local, bidirectional tissue correspondence refinement."""

from .features import structural_feature
from .point import MicroMorphAlign, refine_point
from .plotting import plot_match


def read_image(*args, **kwargs):
    """Load TIFF through the bundled MorphAlign implementation."""
    from morphalign import read_image as _read
    return _read(*args, **kwargs)


def register(*args, **kwargs):
    """Fit a global registration using bundled MorphAlign (requires torch)."""
    from morphalign import register as _register
    return _register(*args, **kwargs)

__version__ = "0.1.0"
__all__ = ["read_image", "register", "structural_feature", "MicroMorphAlign",
           "refine_point", "plot_match"]
