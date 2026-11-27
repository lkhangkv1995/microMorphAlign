"""Training-free 5-DOF registration of REAL H&E to autofluorescence images.

Reference H&E -> moving AF pixel mapping:
 q = c_AF + base_scale * R(phi) @ diag(sx,sy) @ (p-c_HE) + (tx,ty)
where base_scale = max(AF.shape[:2])/max(HE.shape[:2]).

Unlike synthetic demo, no known canvas offset, original AF, or ground truth is used.
The same code handles different sized, different-aspect-ratio source images.

Dependencies: numpy scipy opencv-python torch matplotlib tifffile (recommended).
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json
import time

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.stats import qmc
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2hed

PARAMS = ('scale_x', 'scale_y', 'angle_deg', 'translate_x', 'translate_y')


def read_image(path: str | Path):
    """Read PNG/JPEG/TIFF, including 16-bit and modest multi-channel TIFF images.

    This reads a complete image into RAM, not a pyramidal WSI tile reader.
    For huge WSIs, extract a matching field-of-view or image pyramid first.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Cannot find image: {path}')
    if path.suffix.lower() in ('.tif', '.tiff', '.ome.tif', '.ome.tiff'):
        import tifffile
        arr = tifffile.imread(path)
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f'Could not decode: {path}')
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] <= 256:
            return arr
        if arr.shape[0] <= 256:
            return np.moveaxis(arr, 0, -1)
    raise ValueError(f'Unsupported image shape {arr.shape}: provide 2D or HxWxC TIFF/PNG, not a z-stack')


def _percentile_normalize(ch):
    ch = np.asarray(ch, np.float32)
    finite = np.isfinite(ch)
    if not finite.any():
        return np.zeros_like(ch, dtype=np.float32)
    vals = ch[finite]
    lo, hi = np.percentile(vals, [0.5, 99.7])
    if hi <= lo + 1e-6:
        lo, hi = float(vals.min()), float(vals.max())
    return np.clip((np.nan_to_num(ch, nan=float(lo)) - lo) / max(float(hi - lo), 1e-6), 0, 1)


def preview_rgb(raw, channels=(0, 1, 2)):
    """View transform of raw data; registrations always save original intensities."""
    arr = np.asarray(raw)
    if arr.ndim == 2:
        normalized = _percentile_normalize(arr)
        return np.repeat((normalized * 255).astype(np.uint8)[..., None], 3, axis=-1)
    if arr.shape[2] in (3, 4) and arr.dtype == np.uint8:
        return arr[..., :3].copy()
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    else:
        idx = [min(max(0, int(c)), arr.shape[2]-1) for c in channels]
        arr = arr[..., idx]
    return (np.stack([_percentile_normalize(arr[..., c]) for c in range(3)], -1) * 255).astype(np.uint8)


def signal_image(raw, modality, channel='auto'):
    """One optical image used to generate cross-modal structural features."""
    arr = preview_rgb(raw).astype(np.float32) / 255.0
    if modality == 'he':
        # Bright-field grayscale is the least assumption-heavy real-tissue baseline.
        val = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return val
    raw = np.asarray(raw)
    if channel == 'auto':
        # Use maximum of AF channels to retain signal across different fluorophores.
        return arr.max(axis=2)
    if channel == 'gray':
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    channel = int(channel)
    if raw.ndim == 2:
        return _percentile_normalize(raw)
    if not (0 <= channel < raw.shape[-1]):
        raise ValueError(f'AF channel {channel} out of range for {raw.shape[-1]} channels')
    return _percentile_normalize(raw[..., channel])


def _legacy_tissue_mask(raw, modality, mode='auto'):
    """Image-only approximate observable tissue mask (not a ground-truth mask)."""
    h, w = raw.shape[:2]
    if mode == 'none':
        return np.ones((h, w), dtype=np.float32)
    if mode != 'auto':
        raise ValueError('mask_mode must be "auto" or "none"')
    if modality == 'he':
        rgb = preview_rgb(raw).astype(np.float32) / 255.0
        val = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = ((val < .94) | ((hsv[..., 1] > .10) & (val < .99))).astype(np.uint8)
    else:
        val = signal_image(raw, 'af')
        # A conservative threshold discards true-black padding, but may discard
        # dim AF tissue. Disable with mask_mode='none' when needed.
        mask = (val > .018).astype(np.uint8)
    # Morphological operations at 0.3–1% of short image dimension.
    k = max(3, min(41, int(round(min(h, w) * .006)) | 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mask = binary_fill_holes(mask > 0).astype(np.uint8)
    # Don't allow a pathological segmentation to make optimization undefined.
    if mask.mean() < .008:
        return np.ones((h, w), dtype=np.float32)
    return mask.astype(np.float32)


def tissue_mask(raw, modality, mode='auto'):
    """Automatically choose scanner-aware masks for H&E and 2D nuclear AF."""
    if mode == 'none':
        return np.ones(raw.shape[:2], dtype=np.float32)
    if mode in ('he_stain','dapi_texture') or (mode=='auto' and
             (modality=='he' or (modality=='af' and np.asarray(raw).ndim==2))):
        if mode=='dapi_texture' or (mode=='auto' and modality=='af'):
            dummy=np.full((16,16,3),255,dtype=np.uint8)
            return _tissue_masks_native(dummy,raw)[1]
        dummy=np.zeros((16,16),dtype=np.uint16)
        return _tissue_masks_native(raw,dummy)[0]
    return _legacy_tissue_mask(raw, modality, mode)


def _edge_features(img, sigma=1.5):
    img = np.asarray(img, np.float32)
    # Two scales: fine tissue boundaries and broad tissue structures.
    out = []
    for multiplier in (1., 3.):
        sm = cv2.GaussianBlur(img, (0, 0), sigmaX=max(.6, sigma * multiplier))
        dx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(dx, dy)
        scale = max(float(np.percentile(magnitude, 99.5)), 1e-5)
        out.append(np.clip(magnitude / scale, 0, 1).astype(np.float32))
    return np.stack(out, axis=0)


def _proxy_features(raw, modality, sigma=2.0):
    """Optional tissue-specific proxy inspired by the synthetic experiment.

    Real H&E/AF modality channels can differ drastically. This option is NOT
    universally appropriate and should be visually inspected before use.
    """
    rgb = preview_rgb(raw).astype(np.float32) / 255.
    if modality == 'he':
        x = np.maximum(rgb[..., 2] - rgb[..., 0] + .012, 0)
    else:
        x = np.maximum(rgb[..., 1] - .42 * rgb[..., 2] - .2 * rgb[..., 0] - .025, 0)
    smooth = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma * 3)
    edge = _edge_features(smooth, sigma)
    magnitude = np.clip(smooth / max(np.percentile(smooth, 99.5), 1e-5), 0, 1)
    return np.concatenate([magnitude[None].astype(np.float32), edge], axis=0)


def _resize_channels(array, size, interpolation=cv2.INTER_AREA):
    """Resize arbitrary channel-last optical images; OpenCV has <=4-channel limits."""
    array=np.asarray(array)
    if array.ndim==2 or array.shape[-1]<=4:
        out=cv2.resize(array,size,interpolation=interpolation)
        if array.ndim==3 and out.ndim==2:
            out=out[...,None]
        return out
    return np.stack([cv2.resize(array[...,i],size,interpolation=interpolation)
                     for i in range(array.shape[-1])],axis=-1)


def _warp_channels(array, M, size, interpolation=cv2.INTER_LINEAR):
    """Warp any positive number of channels with identical sampling coordinates."""
    array=np.asarray(array)
    if array.ndim==2 or array.shape[-1]<=4:
        out=cv2.warpAffine(array,M[:2],size,flags=interpolation|cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        if array.ndim==3 and out.ndim==2:
            out=out[...,None]
        return out
    return np.stack([cv2.warpAffine(array[...,i],M[:2],size,
                   flags=interpolation|cv2.WARP_INVERSE_MAP,
                   borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                   for i in range(array.shape[-1])],axis=-1)


def _resized_raw(raw, target_max):
    H, W = raw.shape[:2]
    factor = min(1., float(target_max) / max(H, W))
    hh, ww = max(16, round(H * factor)), max(16, round(W * factor))
    if (hh, ww) == (H, W):
        return raw
    return _resize_channels(raw, (ww, hh), interpolation=cv2.INTER_AREA)


def _thumbnail(arr, side=1300):
    h, w = arr.shape[:2]
    if max(h, w) <= side:
        return arr.copy()
    fac = side / max(h, w)
    return _resize_channels(arr, (max(2, round(w*fac)), max(2, round(h*fac))),
                      interpolation=cv2.INTER_AREA)


def _clean_mask(binary, kernel=11):
    binary = binary.astype(np.uint8)
    k = max(3, int(kernel) | 1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary,8)
    if n > 1:
        largest = stats[1:,cv2.CC_STAT_AREA].max()
        keep = np.zeros(n,np.uint8)
        keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= max(70, largest*.003)).astype('uint8')
        binary = keep[labels]
    binary = binary_fill_holes(binary>0).astype(np.uint8)
    return binary


def _tissue_masks_native(he, dapi, thumb=1300):
    he_s = _thumbnail(he,thumb)
    d_s = _thumbnail(dapi,thumb)
    he_rgb = preview_rgb(he_s).astype(np.float32)/255
    hsv = cv2.cvtColor(he_rgb,cv2.COLOR_RGB2HSV)
    sat = hsv[...,1]; light = hsv[...,2]
    # White mounting background is not tissue.  Saturation is quite high in
    # this uploaded H&E while true white has almost zero saturation.
    hm = _clean_mask((sat>.075) & (light<.995),kernel=13)

    # A DAPI scan can contain uniform nonzero regions (255 in this case)
    # outside the actual tissue.  Distinguish local nuclear texture from those
    # flat plateaus, including possible 0-valued black padding.
    d = np.asarray(d_s,np.float32)
    d_sm = cv2.GaussianBlur(d,(0,0),1.1)
    local_mu = cv2.GaussianBlur(d_sm,(0,0),6)
    local_mu2 = cv2.GaussianBlur(d_sm*d_sm,(0,0),6)
    local_std = np.sqrt(np.maximum(local_mu2-local_mu*local_mu,0))
    texture = cv2.GaussianBlur(local_std,(0,0),3)
    # Exclude flat acquisition background when it is detectable in the borders.
    borders = np.concatenate([d[:8,:].ravel(),d[-8:,:].ravel(),
                              d[:, :8].ravel(),d[:,-8:].ravel()])
    border_mode = float(np.median(borders[borders>0])) if (borders>0).any() else 0
    high = 0.7 * border_mode if border_mode>120 else np.inf
    tissue = (texture > 11) & (local_mu>5) & (local_mu < high)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dm = cv2.morphologyEx(tissue.astype(np.uint8),cv2.MORPH_CLOSE,k1)
    dm = cv2.morphologyEx(dm,cv2.MORPH_OPEN,k2)
    n,labels,stats,_ = cv2.connectedComponentsWithStats(dm,8)
    if n>1:
        max_area = stats[1:,cv2.CC_STAT_AREA].max()
        keep = np.zeros(n,np.uint8)
        keep[1:] = (stats[1:,cv2.CC_STAT_AREA]>=max(150,max_area*.003)).astype('uint8')
        dm = keep[labels]
    # Fill only modest cellular holes, never an entire exterior background
    # region encircled by a scanner-edge stripe.
    holes = binary_fill_holes(dm>0) & ~(dm>0)
    n,labels,stats,_=cv2.connectedComponentsWithStats(holes.astype(np.uint8),8)
    for idx in range(1,n):
        if stats[idx,cv2.CC_STAT_AREA] < 700:
            dm[labels==idx]=1

    hm = cv2.resize(hm.astype('float32'),(he.shape[1],he.shape[0]),interpolation=cv2.INTER_NEAREST)
    dm = cv2.resize(dm.astype('float32'),(dapi.shape[1],dapi.shape[0]),interpolation=cv2.INTER_NEAREST)
    return hm,dm


def real_nuclei_features(raw,modality,mask,native_sigma=1.6):
    """Nucleus-rich morphology mapped to comparable feature vectors.

    H&E: positive hematoxylin optical density from HED deconvolution.
    DAPI: log-compressed fluorescence, with uniform padded background removed.
    Multi-scale Gaussian density maps are less dependent on pixelwise cell
    correspondence (e.g. between serial sections) than raw intensity.
    """
    if modality == 'he':
        rgb = preview_rgb(raw).astype(np.float32)/255
        hemi = rgb2hed(rgb)[...,0].astype(np.float32)
        hemi = np.maximum(hemi,0)
        f = hemi
    else:
        a = np.asarray(raw)
        if a.ndim==3: a=a[...,0]
        a = np.maximum(a.astype(np.float32),0)
        f = np.log1p(a)
    mask = np.asarray(mask,np.float32)
    vals = f[mask>.5]
    lo,hi = np.percentile(vals,[7,99.6]) if len(vals)>100 else np.percentile(f,[7,99.6])
    f = np.clip((f-lo)/max(1e-6,hi-lo),0,1)*mask
    # Blur at matched *relative* tissue scales; two smooth channels represent
    # local nuclear density and a third captures spatial tissue texture.
    sm1=cv2.GaussianBlur(f,(0,0),sigmaX=1.4)
    sm2=cv2.GaussianBlur(f,(0,0),sigmaX=4)
    sm3=cv2.GaussianBlur(f,(0,0),sigmaX=9)
    var=np.maximum(cv2.GaussianBlur(f*f,(0,0),sigmaX=3)-
                   cv2.GaussianBlur(f,(0,0),sigmaX=3)**2,0)
    tex=np.sqrt(var)
    # Make feature weights naturally comparable across contrast and bit-depth
    out=[sm1,sm2,sm3,tex]
    for i,x in enumerate(out):
        q=np.percentile(x[mask>.5],[5,99]) if (mask>.5).sum()>100 else (0,1)
        out[i]=np.clip((x-q[0])/max(float(q[1]-q[0]),1e-5),0,1).astype(np.float32)*mask
    return np.stack(out,0)



def _generic_tissue_mask(raw):
    """Foreground estimated from border-vs-interior signal, channel independently.

    This is a heuristic for brightfield and fluorescence fields, not a tissue label.
    """
    arr=np.asarray(raw)
    if arr.ndim==2: arr=arr[...,None]
    h,w=arr.shape[:2]
    target=max(96,min(768,max(h,w)))
    small=_resize_channels(arr,(max(8,int(round(w*target/max(h,w)))),max(8,int(round(h*target/max(h,w))))),interpolation=cv2.INTER_AREA)
    if small.ndim==2: small=small[...,None]
    energy=np.zeros(small.shape[:2],np.float32)
    for c in range(small.shape[-1]):
        x=small[...,c].astype(np.float32)
        border=np.concatenate((x[0],x[-1],x[:,0],x[:,-1]))
        bg=float(np.median(border))
        dev=np.abs(x-bg)
        lo,hi=np.percentile(dev,[10,99.5])
        if hi>lo+1e-6: energy=np.maximum(energy,np.clip((dev-lo)/(hi-lo),0,1))
    u8=np.uint8(np.clip(energy*255,0,255))
    if u8.max()<10: return np.ones((h,w),np.float32)
    _,binary=cv2.threshold(u8,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    binary=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    binary=cv2.dilate(binary,np.ones((5,5),np.uint8),iterations=1)
    if binary.mean()<4:binary[:]=255  # insufficient contrast: do not hallucinate a mask
    return cv2.resize(binary.astype(np.float32)/255,(w,h),interpolation=cv2.INTER_NEAREST)


def _multichannel_structure(raw, sigma=1.5, channels=None):
    """Channel-agnostic structural edge energy for C>=1; no learned weights.

    Each selected optical channel is percentile normalized independently;
    the top-quantile edge energy is pooled across channels. Different channel
    counts are permitted. Fine-scale edges are smoothed for cross-modality fit.
    """
    arr=np.asarray(raw)
    if arr.ndim==2:arr=arr[...,None]
    if channels is None or channels=='auto': channels=range(arr.shape[-1])
    channels=[int(i) for i in channels]
    if not channels or any(i<0 or i>=arr.shape[-1] for i in channels):
        raise ValueError('Invalid feature channel indices')
    maps=[]
    for j in channels:
        x=_percentile_normalize(arr[...,j]).astype(np.float32)
        x=cv2.GaussianBlur(x,(0,0),max(.5,sigma))
        gx=cv2.Sobel(x,cv2.CV_32F,1,0,ksize=3)
        gy=cv2.Sobel(x,cv2.CV_32F,0,1,ksize=3)
        edge=cv2.magnitude(gx,gy)
        peak=np.percentile(edge,99.5)
        maps.append(np.clip(edge/max(float(peak),1e-6),0,1))
    # An edge shared by several channels is not required; median pooling
    # would suppress nuclear-channel-only structure.
    combined=np.max(np.stack(maps,axis=0),axis=0)
    combined=cv2.GaussianBlur(combined,(0,0),max(1.,sigma*2))
    return combined[None].astype(np.float32)


def feature_pyramid(he, af, levels=(96, 160, 256), mode='edges', af_channel='auto',
                    he_mask_mode='auto', af_mask_mode='auto', device='cpu'):
    """Precompute feature maps and their masks. Only image content is used.

    Each image is independently resized while preserving aspect ratio. Grid
    coordinates always map to the ORIGINAL H&E and AF pixel coordinate frames.
    """
    if mode == 'multichannel':
        mask_he = _generic_tissue_mask(he) if he_mask_mode!='none' else np.ones(he.shape[:2],np.float32)
        mask_af = _generic_tissue_mask(af) if af_mask_mode!='none' else np.ones(af.shape[:2],np.float32)
        pyramid=[]
        for side in levels:
            he_s=_resized_raw(he,side)
            af_s=_resized_raw(af,max(side,round(side*max(af.shape[:2])/max(he.shape[:2]))))
            if max(af_s.shape[:2])>2*side:af_s=_resized_raw(af,2*side)
            ref_f=_multichannel_structure(he_s,sigma=1.1)
            mov_f=_multichannel_structure(af_s,sigma=1.1,channels=af_channel)
            rm=cv2.resize(mask_he,(he_s.shape[1],he_s.shape[0]),interpolation=cv2.INTER_AREA)
            mm=cv2.resize(mask_af,(af_s.shape[1],af_s.shape[0]),interpolation=cv2.INTER_AREA)
            pyramid.append({'ref':torch.from_numpy(ref_f[None]).to(device),
                            'mov':torch.from_numpy(mov_f[None]).to(device),
                            'ref_mask':torch.from_numpy(rm[None,None].astype(np.float32)).to(device),
                            'mov_mask':torch.from_numpy(mm[None,None].astype(np.float32)).to(device),
                            'side':int(side)})
        return pyramid
    if mode == 'nuclear_dapi':
        mask_he,mask_af = _tissue_masks_native(he,af)
        if he_mask_mode == 'none':mask_he[:]=1
        if af_mask_mode == 'none':mask_af[:]=1
        pyramid=[]
        for side in levels:
            he_s=_resized_raw(he,side)
            af_s=_resized_raw(af,max(side,round(side*max(af.shape[:2])/max(he.shape[:2]))))
            if max(af_s.shape[:2])>2*side:af_s=_resized_raw(af,2*side)
            hm=cv2.resize(mask_he,(he_s.shape[1],he_s.shape[0]),interpolation=cv2.INTER_AREA)
            dm=cv2.resize(mask_af,(af_s.shape[1],af_s.shape[0]),interpolation=cv2.INTER_AREA)
            rf=real_nuclei_features(he_s,'he',hm)
            mf=real_nuclei_features(af_s,'dapi',dm)
            pyramid.append({'ref':torch.from_numpy(rf[None]).to(device),
                            'mov':torch.from_numpy(mf[None]).to(device),
                            'ref_mask':torch.from_numpy(hm[None,None].astype(np.float32)).to(device),
                            'mov_mask':torch.from_numpy(dm[None,None].astype(np.float32)).to(device),
                            'side':int(side)})
        return pyramid
    mask_he = tissue_mask(he, 'he', he_mask_mode)
    mask_af = tissue_mask(af, 'af', af_mask_mode)
    pyramid = []
    for side in levels:
        he_s = _resized_raw(he, side)
        af_s = _resized_raw(af, max(side, round(side * max(af.shape[:2])/max(he.shape[:2]))))
        # The moving feature footprint need not be the same pixel dimensions.
        # Cap it to ~2x reference pyramid resolution for large black backgrounds.
        if max(af_s.shape[:2]) > 2 * side:
            af_s = _resized_raw(af, 2 * side)
        if mode == 'epithelial_proxy':
            ref_f = _proxy_features(he_s, 'he')
            mov_f = _proxy_features(af_s, 'af')
        elif mode == 'edges':
            ref_f = _edge_features(signal_image(he_s, 'he'), sigma=1.25)
            mov_f = _edge_features(signal_image(af_s, 'af', af_channel), sigma=1.25)
        else:
            raise ValueError("mode='edges' or 'epithelial_proxy'")
        h0, w0 = he_s.shape[:2]
        h1, w1 = af_s.shape[:2]
        rm = cv2.resize(mask_he, (w0, h0), interpolation=cv2.INTER_AREA)
        mm = cv2.resize(mask_af, (w1, h1), interpolation=cv2.INTER_AREA)
        pyramid.append({
            'ref': torch.from_numpy(ref_f).unsqueeze(0).to(device),
            'mov': torch.from_numpy(mov_f).unsqueeze(0).to(device),
            'ref_mask': torch.from_numpy(rm.astype(np.float32))[None,None].to(device),
            'mov_mask': torch.from_numpy(mm.astype(np.float32))[None,None].to(device),
            'side': side,
        })
    return pyramid


@dataclass
class RegistrationConfig:
    # Effective scale is relative to original image max-size ratio.
    scale_x: tuple[float,float] = (.35, 2.8)
    scale_y: tuple[float,float] = (.35, 2.8)
    angle_deg: tuple[float,float] = (-180., 180.)
    # Translations expressed as FRACTIONS of maximum AF image dimension.
    translate_fraction: tuple[float,float] = (-.65, .65)
    min_coverage: float = .50
    coverage_penalty: float = 8.0
    silhouette_weight: float = .20
    prior_weight: float = .005
    feature_mode: str = 'edges'
    af_channel: str | int = 'auto'
    he_mask_mode: str = 'auto'
    af_mask_mode: str = 'auto'
    pyramid_levels: tuple[int,...] = (96,160,256)
    seed: int = 42
    sobol_samples: int = 1024
    exploration_rounds: int = 3
    proposals_per_round: int = 256
    top_k: int = 6
    polish_steps: int = 45
    adam_iterations: tuple[int,...] = (180, 140, 100)
    device: str = 'cpu'


def _z_bounds(cfg):
    return np.array([
        np.log(cfg.scale_x), np.log(cfg.scale_y),
        np.deg2rad(cfg.angle_deg), cfg.translate_fraction,
        cfg.translate_fraction
    ], np.float32)


def _z_to_params(z, he_hw, af_hw):
    z = np.asarray(z, float)
    m = max(af_hw)
    return dict(scale_x=float(np.exp(z[0])), scale_y=float(np.exp(z[1])),
                angle_deg=float(np.rad2deg(z[2])),
                translate_x=float(z[3] * m), translate_y=float(z[4] * m))


def _params_to_z(p, af_hw):
    m = max(af_hw)
    return np.array([np.log(p['scale_x']), np.log(p['scale_y']),
                     np.deg2rad(p['angle_deg']), p['translate_x']/m,
                     p['translate_y']/m], np.float32)


def affine_matrix(params, he_hw, af_hw):
    """3x3 forward *reference H&E coordinates -> moving AF coordinates* matrix."""
    h,w = map(int,he_hw)
    mh,mw = map(int,af_hw)
    c0 = np.array([(w-1)/2.,(h-1)/2.])
    c1 = np.array([(mw-1)/2.,(mh-1)/2.])
    a = np.deg2rad(params['angle_deg'])
    factor = max(mh,mw)/max(h,w)
    # OpenCV pixel coordinates, positive angle rotates visually counterclockwise.
    R = np.array([[np.cos(a), np.sin(a)],[-np.sin(a),np.cos(a)]])
    A = factor * R @ np.diag([params['scale_x'],params['scale_y']])
    t = c1 + np.array([params['translate_x'], params['translate_y']]) - A@c0
    M = np.eye(3)
    M[:2,:2],M[:2,2] = A,t
    return M


def _grids(z, he_hw, af_hw, out_hw):
    """Torch differentiable mapping of reference grid -> native moving coords."""
    if z.ndim == 1:
        z = z[None,:]
    h,w=he_hw; mh,mw=af_hw
    oh,ow=out_hw
    xx,yy = torch.meshgrid(
        torch.linspace(0,w-1,ow,device=z.device,dtype=z.dtype),
        torch.linspace(0,h-1,oh,device=z.device,dtype=z.dtype),indexing='xy')
    # torch meshgrid(indexing='xy') returns (oh, ow); xx/yy explicit below.
    dx = xx - (w-1)/2.
    dy = yy - (h-1)/2.
    fac = max(mh,mw)/max(h,w)
    sx,sy = z[:,0].exp(),z[:,1].exp()
    co,si = z[:,2].cos(),z[:,2].sin()
    mx = ((co*sx*fac)[:,None,None]*dx[None] +
          (si*sy*fac)[:,None,None]*dy[None] +
          ((mw-1)/2. + z[:,3]*max(mh,mw))[:,None,None])
    my = ((-si*sx*fac)[:,None,None]*dx[None] +
          (co*sy*fac)[:,None,None]*dy[None] +
          ((mh-1)/2. + z[:,4]*max(mh,mw))[:,None,None])
    return torch.stack([mx*(2./max(1,mw-1))-1,my*(2./max(1,mh-1))-1],-1)


def _objective(z, level, he_hw, af_hw, cfg):
    if z.ndim==1:z=z[None]
    g = _grids(z,he_hw,af_hw,level['ref'].shape[-2:])
    n = z.shape[0]
    mov = F.grid_sample(level['mov'].expand(n,-1,-1,-1),g,
                        mode='bilinear',padding_mode='zeros',align_corners=True)
    mm = F.grid_sample(level['mov_mask'].expand(n,-1,-1,-1),g,
                       mode='bilinear',padding_mode='zeros',align_corners=True)
    valid_xy = (g[...,0].abs()<=1) & (g[...,1].abs()<=1)
    mask = (mm * level['ref_mask'] * valid_xy[:,None].float()).clamp(0,1)
    c = level['ref'].shape[1]
    r = level['ref'].expand(n,-1,-1,-1)
    den = mask.sum((-2,-1),keepdim=True).clamp_min(1.)
    mean_r = (mask*r).sum((-2,-1),keepdim=True)/den
    mean_m = (mask*mov).sum((-2,-1),keepdim=True)/den
    rr = r-mean_r; mmv=mov-mean_m
    ncc = (mask*rr*mmv).sum((-2,-1)) / (
        ((mask*rr.square()).sum((-2,-1)) *
          (mask*mmv.square()).sum((-2,-1))).clamp_min(1e-7)).sqrt()
    ncc = ncc.mean(1)
    coverage = (mask.sum((-2,-1))/level['ref_mask'].sum().clamp_min(1.)).flatten()
    penalty = cfg.coverage_penalty * F.relu(cfg.min_coverage - coverage).square()
    # A weak silhouette/Dice constraint discourages aligning only a small patch.
    # Set silhouette_weight=0 if the images show differing tissue extents.
    shape_agreement = 1. - (2 * (level['ref_mask'] * mm * valid_xy[:,None]).sum((-2,-1)) /
                             (level['ref_mask'].sum()+ (mm * valid_xy[:,None]).sum((-2,-1)) + 1e-6)).flatten()
    penalty = penalty + cfg.silhouette_weight * shape_agreement
    prior = cfg.prior_weight * (z[:,0].square() + z[:,1].square() +
                                .10*z[:,3].square()+.10*z[:,4].square())
    loss = 1 - ncc + penalty + prior
    return loss, ncc.detach(), coverage.detach()


def _batch_scores(points,level,he_hw,af_hw,cfg,batch=32):
    scores = []
    with torch.no_grad():
        for start in range(0,len(points),batch):
            z=torch.as_tensor(points[start:start+batch],device=cfg.device,dtype=torch.float32)
            res,_,_=_objective(z,level,he_hw,af_hw,cfg)
            scores.extend(res.cpu().numpy().tolist())
    return np.array(scores,dtype=np.float64)


def _diverse_indices(z, losses, bounds, n, separation=.08):
    norm = (z - bounds[:,0])/(bounds[:,1]-bounds[:,0]+1e-8)
    chosen=[]
    for i in np.argsort(losses):
        if all(np.linalg.norm(norm[i]-norm[j]) >= separation for j in chosen):
            chosen.append(int(i))
        if len(chosen)>=n:break
    if len(chosen)<n:
        for i in np.argsort(losses):
            if int(i) not in chosen:chosen.append(int(i))
            if len(chosen)>=n:break
    return np.array(chosen,np.int64)


def _adam(pyramid,he_hw,af_hw,cfg,initial,iterations,lr=(.06,.025,.012)):
    zb=_z_bounds(cfg)
    z=torch.nn.Parameter(torch.tensor(initial,dtype=torch.float32,device=cfg.device))
    history=[]
    for stage,(level,iters) in enumerate(zip(pyramid,iterations)):
        if iters<=0:continue
        opt=torch.optim.Adam([z],lr=lr[min(stage,len(lr)-1)])
        for it in range(int(iters)):
            opt.zero_grad()
            losses,ncc,coverage=_objective(z,level,he_hw,af_hw,cfg)
            losses.mean().backward()
            opt.step()
            with torch.no_grad():
                z.clamp_(min=torch.as_tensor(zb[:,0],device=z.device),
                         max=torch.as_tensor(zb[:,1],device=z.device))
            if it % 1 == 0:
                history.append((stage,it,float(losses.mean().item()),
                                float(ncc.mean().item()),float(coverage.mean().item())))
    return z.detach().cpu().numpy(),np.array(history,np.float64)


def _mask_centroid(mask):
    yy,xx=np.nonzero(mask>.3)
    if len(xx)<5: return None
    return np.array([float(xx.mean()),float(yy.mean())])


def _centroid_seeds(mask_he, mask_af, he_hw, af_hw, cfg):
    """Additional unsupervised geometric seeds, not ground truth."""
    r=_mask_centroid(mask_he);m=_mask_centroid(mask_af)
    if r is None or m is None:return []
    h,w=he_hw; mh,mw=af_hw
    c0=np.array([(w-1)/2,(h-1)/2]);c1=np.array([(mw-1)/2,(mh-1)/2])
    fac=max(mh,mw)/max(h,w)
    seeds=[]
    for theta in [0,45,90,135,180,-45,-90,-135]:
        for scale in [0.6,1.,1.5]:
            a=np.deg2rad(theta)
            R=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
            tx_ty = (m-c1-fac*scale*R@(r-c0))/max(mh,mw)
            z=np.array([np.log(scale),np.log(scale),a,*tx_ty],np.float32)
            bounds=_z_bounds(cfg)
            seeds.append(np.clip(z,bounds[:,0],bounds[:,1]))
    return seeds



def _rectangle_seeds(mask_he, mask_af, he_hw, af_hw, cfg):
    """Rough orientation/scale estimates from tissue silhouettes.

    A useful starting point for fully visible tissue, but not authoritative if
    the AF and H&E crops show different tissue regions. Always optimized and
    ranked using the same image-based objective as other seeds.
    """
    def bbox_info(mask):
        binmask=(mask > .35).astype(np.uint8)
        if binmask.sum() < 20: return None
        cnts,_=cv2.findContours(binmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:return None
        # A union of tissue fragments better represents a fragmented sample.
        coords=np.argwhere(binmask>0)
        y0,x0=coords.min(0);y1,x1=coords.max(0)
        return (np.array([(x0+x1)/2.,(y0+y1)/2.]),float(x1-x0+1),float(y1-y0+1),
                cv2.minAreaRect(np.c_[coords[:,1],coords[:,0]].astype(np.float32)))
    ref=bbox_info(mask_he); mov=bbox_info(mask_af)
    if ref is None or mov is None:return []
    center_ref,width_ref,height_ref,_=ref
    center_mov,_,_,box=mov
    edges=cv2.boxPoints(box)
    edge_a=edges[1]-edges[0];edge_b=edges[2]-edges[1]
    vectors=[edge_a, -edge_a, edge_b, -edge_b]
    fac=max(af_hw)/max(he_hw)
    center0=np.array([(he_hw[1]-1)/2.,(he_hw[0]-1)/2.])
    center1=np.array([(af_hw[1]-1)/2.,(af_hw[0]-1)/2.])
    bounds=_z_bounds(cfg)
    seeds=[]
    for v in vectors:
        length=np.linalg.norm(v)
        other_len=np.linalg.norm(edge_b if np.isclose(length,np.linalg.norm(edge_a)) else edge_a)
        # Degenerate square boxes: both potential side assignments tried.
        for length_x,length_y in [(length,other_len),(other_len,length)]:
            sx=max(.001,length_x/(width_ref*fac))
            sy=max(.001,length_y/(height_ref*fac))
            angle=np.arctan2(-v[1],v[0])
            R=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
            tv=(center_mov-center1-fac*R@np.diag([sx,sy])@(center_ref-center0))/max(af_hw)
            zz=np.array([np.log(sx),np.log(sy),angle,*tv],np.float32)
            seeds.append(np.clip(zz,bounds[:,0],bounds[:,1]))
    return seeds

def register_images(he,af,cfg: Optional[RegistrationConfig]=None,
                    initialization:Optional[dict]=None,progress=print):
    """Run global Sobol burn-in + multi-start shortlist + Adam; return results.

    No true registration parameters or correspondences are given to the algorithm.
    For real data, TRE cannot be computed without user-provided landmarks.
    """
    if cfg is None:cfg=RegistrationConfig()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    start=time.perf_counter()
    he_hw=he.shape[:2];af_hw=af.shape[:2]
    pyr=feature_pyramid(he,af,levels=cfg.pyramid_levels,
                        mode=cfg.feature_mode,af_channel=cfg.af_channel,
                        he_mask_mode=cfg.he_mask_mode,af_mask_mode=cfg.af_mask_mode,
                        device=cfg.device)
    zb=_z_bounds(cfg); lo,hi=zb[:,0],zb[:,1]
    sampler=qmc.Sobol(d=5,scramble=True,seed=cfg.seed)
    # random_base2 avoids Sobol imbalance warnings if N is a power of 2.
    n=int(cfg.sobol_samples)
    if n>0:
        samples=sampler.random_base2(int(np.ceil(np.log2(n))))[:n].astype(np.float32)
        z=lo+(hi-lo)*samples
    else:
        z=np.empty((0,5),np.float32)
    identity=np.zeros(5,np.float32)
    z=np.vstack([identity,z])
    if initialization is not None:
        z=np.vstack([z,_params_to_z(initialization,af_hw)])
    # Tissue geometry seeds MUST be computed in the ORIGINAL coordinate grids.
    # The pyramid masks are resized and cannot be interpreted as native pixels.
    refmask=_generic_tissue_mask(he) if cfg.feature_mode=='multichannel' else tissue_mask(he,'he',cfg.he_mask_mode)
    movmask=_generic_tissue_mask(af) if cfg.feature_mode=='multichannel' else tissue_mask(af,'af',cfg.af_mask_mode)
    z=np.vstack([z,*_centroid_seeds(refmask,movmask,he_hw,af_hw,cfg),
                 *_rectangle_seeds(refmask,movmask,he_hw,af_hw,cfg)])
    z=np.clip(z,lo,hi).astype(np.float32)
    losses=_batch_scores(z,pyr[0],he_hw,af_hw,cfg)
    if progress:progress(f'Scored {len(z):,} global candidates | best coarse loss {losses.min():.4f}')
    rng=np.random.default_rng(cfg.seed+1)
    for rnd in range(cfg.exploration_rounds):
        elite=_diverse_indices(z,losses,zb,min(18,len(z)),separation=.11)
        parents=z[rng.choice(elite,size=cfg.proposals_per_round)]
        radius=[.17,.09,.045,.025][min(rnd,3)]
        proposed=np.clip(parents+rng.normal(size=parents.shape).astype(np.float32)*
                         (hi-lo)*radius,lo,hi).astype(np.float32)
        more_losses=_batch_scores(proposed,pyr[0],he_hw,af_hw,cfg)
        z=np.vstack([z,proposed]);losses=np.concatenate([losses,more_losses])
        if progress:progress(f'Round {rnd+1}/{cfg.exploration_rounds}: best loss {losses.min():.4f}')
    best_ids=_diverse_indices(z,losses,zb,min(cfg.top_k,len(z)),separation=.08)
    if progress:progress(f'Polishing {len(best_ids)} candidates with short Adam runs…')
    winners=[];refined=[]
    for idx in best_ids:
        if cfg.polish_steps:
            npolish=len(pyr)
            steps=[cfg.polish_steps//2,cfg.polish_steps-cfg.polish_steps//2]+[0]*(npolish-2)
            zp,_=_adam(pyr,he_hw,af_hw,cfg,z[idx],steps)
        else: zp=z[idx]
        winners.append(zp)
    winners=np.array(winners,np.float32)
    selection_level=min(1,len(pyr)-1)
    refined=_batch_scores(winners,pyr[selection_level],he_hw,af_hw,cfg)
    init=winners[np.argmin(refined)]
    if progress:progress('Final coarse-to-fine Adam optimization…')
    estimated_z,history=_adam(pyr,he_hw,af_hw,cfg,init,cfg.adam_iterations)
    estimated=_z_to_params(estimated_z,he_hw,af_hw)
    M=affine_matrix(estimated,he_hw,af_hw)
    # Determine final coarse-space self-consistency, NOT ground-truth accuracy.
    fin,ncc,cov=_objective(torch.tensor(estimated_z,device=cfg.device),pyr[-1],he_hw,af_hw,cfg)
    result=dict(parameters=estimated,matrix_he_to_af=M,
                matrix_af_to_he=np.linalg.inv(M),
                initial_parameters=_z_to_params(init,he_hw,af_hw),
                history=history,coarse_candidate_scores=losses,
                coarse_candidate_z=z,shortlist_z=winners,
                shortlist_scores=refined,
                final_loss=float(fin.item()),final_ncc=float(ncc.item()),
                reference_coverage=float(cov.item()),elapsed_seconds=time.perf_counter()-start,
                he_shape=list(he.shape),af_shape=list(af.shape),config=cfg.__dict__.copy())
    if progress:progress(f'Done | NCC {result["final_ncc"]:.3f} | '
                         f'coverage {result["reference_coverage"]:.1%} | '
                         f'{result["elapsed_seconds"]:.1f}s')
    return result


def warp_af_to_he(raw_af,result,he_hw,interpolation=cv2.INTER_LINEAR):
    """Resample ORIGINAL AF intensities into H&E pixel coordinates."""
    h,w=map(int,he_hw)
    M=np.asarray(result['matrix_he_to_af'],np.float64)
    return _warp_channels(raw_af,M,(w,h),interpolation=interpolation)


def mapped_af_valid_mask(af_hw,result,he_hw):
    ones=np.ones(af_hw,dtype=np.uint8)
    return warp_af_to_he(ones,result,he_hw,interpolation=cv2.INTER_NEAREST)


def save_result(output_dir,he,af,result,prefix='registration'):
    import tifffile
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    registered=warp_af_to_he(af,result,he.shape[:2])
    # Save full original bit-depth, potentially multi-channel, not preview RGB.
    tifffile.imwrite(output/f'{prefix}_af_registered.tif',registered)
    np.save(output/f'{prefix}_he_to_af_3x3.npy',result['matrix_he_to_af'])
    np.save(output/f'{prefix}_af_to_he_3x3.npy',result['matrix_af_to_he'])
    np.save(output/f'{prefix}_valid_mask.npy',mapped_af_valid_mask(af.shape[:2],result,he.shape[:2]))
    simple={k:v for k,v in result.items() if k not in
            ('history','coarse_candidate_scores','coarse_candidate_z','shortlist_z','shortlist_scores',
             'matrix_he_to_af','matrix_af_to_he')}
    (output/f'{prefix}_parameters.json').write_text(json.dumps(simple,indent=2))
    np.savez_compressed(output/f'{prefix}_optimization.npz',
                        history=result['history'],
                        coarse_candidate_scores=result['coarse_candidate_scores'],
                        coarse_candidate_z=result['coarse_candidate_z'],
                        shortlist_z=result['shortlist_z'],
                        shortlist_scores=result['shortlist_scores'])
    return output


def register_he_dapi(he, dapi, config=None, initialization=None, progress=print):
    """Run five-parameter H&E–DAPI registration, with nuclear-density matching."""
    if config is None:
        config=RegistrationConfig(
            feature_mode='nuclear_dapi',pyramid_levels=(128,256,384,640),
            adam_iterations=(150,120,95,70),sobol_samples=1024,
            exploration_rounds=3,proposals_per_round=256,top_k=8,
            polish_steps=45,coverage_penalty=7.,silhouette_weight=.4,
            min_coverage=.65,prior_weight=.002,seed=31)
    return register_images(he,dapi,config,initialization=initialization,progress=progress)
