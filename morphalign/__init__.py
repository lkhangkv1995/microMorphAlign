"""Generic public API for 2D multichannel registration.

image1 is the reference; image2 is resampled onto image1's pixel grid.
Supports grayscale (H,W) and channel-last (H,W,C) arrays.
"""
from .affine import read_image, preview_rgb
from .pipeline import run_registration, export_result


def register(image1, image2, *, deformable=True, channels='auto',
             sobol_samples=256, exploration_rounds=2,
             affine_iterations=(75,55,35), deform_iterations=(55,55),
             max_displacement=30.0, device='cpu', progress=None):
    """Register image2 onto image1; no fixed channel count or stain labels.

    `channels` selects optional 0-based channels of image2 for feature matching.
    The saved registered image always includes ALL image2 channels.
    """
    return run_registration(image1,image2,feature_mode='multichannel',
        moving_channels=channels,deformable=deformable,
        sobol_samples=sobol_samples,exploration_rounds=exploration_rounds,
        affine_iterations=affine_iterations,deform_iterations=deform_iterations,
        max_displacement=max_displacement,device=device,progress=progress)


__all__=['read_image','preview_rgb','register','export_result']
