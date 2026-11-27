"""Public pipeline for multichannel affine + optional deformable alignment.

Reference coordinates -> moving coordinates; channel dimension is last.
No fluorescent channel is assumed to correspond to an RGB color channel.
"""
from pathlib import Path
import json
import numpy as np
import cv2
import tifffile
from .affine import RegistrationConfig, register_images, warp_af_to_he, preview_rgb
from .deformable import optimize_deformation, warp_native_multichannel, dense_field, field_jacobian


def run_registration(reference, moving, *, feature_mode='multichannel',
                     moving_channels='auto', deformable=True,
                     sobol_samples=256, exploration_rounds=2,
                     affine_iterations=(75, 55, 35), deform_iterations=(55, 55),
                     max_displacement=30.0, device='cpu', progress=None):
    """Accept numpy HxW or HxWxC arrays (C>=1) and preserve moving dtype."""
    reference=np.asarray(reference);moving=np.asarray(moving)
    if reference.ndim not in (2,3) or moving.ndim not in (2,3):
        raise ValueError('Inputs must be HxW or HxWxC images; select one plane from stacks')
    if feature_mode=='nuclear_dapi' and (reference.ndim!=3 or reference.shape[-1]!=3):
        raise ValueError('nuclear_dapi mode expects RGB reference (H&E)')
    cfg=RegistrationConfig(feature_mode=feature_mode,af_channel=moving_channels,
        sobol_samples=int(sobol_samples),exploration_rounds=int(exploration_rounds),
        proposals_per_round=128,top_k=5,polish_steps=24,
        pyramid_levels=(96,160,256),adam_iterations=tuple(affine_iterations),device=device)
    fitted=register_images(reference,moving,cfg,progress=progress)
    registered_affine=warp_af_to_he(moving,fitted,reference.shape[:2])
    result={'affine':fitted,'affine_registered':registered_affine,
            'registered':registered_affine,'deformable':None,
            'feature_mode':feature_mode}
    if deformable:
        h,w=reference.shape[:2]
        ratio=w/max(h,1)
        controls=((max(5,round(8/ max(.5,ratio))),max(5,round(8*ratio))),
                  (max(7,round(12/max(.5,ratio))),max(7,round(12*ratio))))
        df=optimize_deformation(reference,moving,fitted['matrix_he_to_af'],
            levels=(192,320),control_shapes=controls,iterations=tuple(deform_iterations),
            max_displacement=float(max_displacement),device=device,verbose=False,
            feature_mode=feature_mode,af_channel=moving_channels)
        registered=warp_native_multichannel(moving,reference.shape[:2],df)
        result['registered']=registered
        result['deformable']=df
    return result


def export_result(result, reference, moving, output_dir):
    """Export original-depth multichannel TIFF and matrices, full spatial metadata."""
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    tifffile.imwrite(output_dir/'registered_moving.tif',result['registered'],metadata={'axes':'YXC'} if result['registered'].ndim==3 else None)
    tifffile.imwrite(output_dir/'affine_only_moving.tif',result['affine_registered'],metadata={'axes':'YXC'} if result['affine_registered'].ndim==3 else None)
    np.save(output_dir/'reference_to_moving_affine.npy',result['affine']['matrix_he_to_af'])
    np.save(output_dir/'image1_to_image2_affine.npy',result['affine']['matrix_he_to_af'])
    np.save(output_dir/'moving_to_reference_affine.npy',result['affine']['matrix_af_to_he'])
    np.save(output_dir/'image2_to_image1_affine.npy',result['affine']['matrix_af_to_he'])
    metadata=dict(reference_shape=list(reference.shape),moving_shape=list(moving.shape),
        registered_shape=list(result['registered'].shape),moving_dtype=str(moving.dtype),
        feature_mode=result['feature_mode'],affine_parameters=result['affine']['parameters'],
        affine_matrix_reference_to_moving=result['affine']['matrix_he_to_af'].tolist(),
        affine_ncc=result['affine']['final_ncc'],affine_reference_coverage=result['affine']['reference_coverage'],
        deformation_enabled=result['deformable'] is not None)
    if result['deformable'] is not None:
        df=result['deformable']
        np.save(output_dir/'deformation_controls.npy',df['controls'])
        np.save(output_dir/'deformation_history.npy',df['history'])
        metadata['deformation_max_displacement_px']=df['max_displacement']
        metadata['deformation_controls_shape']=list(df['controls'].shape)
        # Dense field is deliberately not included by default to keep exports small.
    (output_dir/'registration.json').write_text(json.dumps(metadata,indent=2))
    return output_dir
