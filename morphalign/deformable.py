"""Smooth, regularized H&E-DAPI deformable refinement after 5-DOF affine registration.

Maps an H&E/reference pixel p to DAPI source M @ (p + u(p)), where u is a
smooth displacement in H&E pixels. The affine M remains fixed. Every field
is defined in H&E pixel coordinates; no hidden reliance on equal image sizes.

Caution: a higher optimized feature NCC is not independent registration proof.
"""
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import tifffile
from . import affine as reg


def _reference_grid(shape, device='cpu'):
    h,w=shape
    yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=device),
                         torch.linspace(-1,1,w,device=device),indexing='ij')
    return torch.stack([xx,yy],dim=-1)[None]


def _upsample_flow(ctrl, output_shape):
    # control grid in FULL resolution H&E pixel units
    return F.interpolate(ctrl,size=output_shape,mode='bicubic',align_corners=True)


def _sample_grid(shape, ref_hw, mov_hw, affine, flow):
    """Return normalized moving AF sampling grid at ref feature-map resolution."""
    oh,ow=shape
    h,w=ref_hw; mh,mw=mov_hw
    yy,xx=torch.meshgrid(torch.linspace(0,h-1,oh,device=flow.device),
                         torch.linspace(0,w-1,ow,device=flow.device),indexing='ij')
    px=xx+flow[0,0]
    py=yy+flow[0,1]
    A=torch.tensor(affine[:2,:2],device=flow.device,dtype=flow.dtype)
    b=torch.tensor(affine[:2,2],device=flow.device,dtype=flow.dtype)
    mx=A[0,0]*px+A[0,1]*py+b[0]
    my=A[1,0]*px+A[1,1]*py+b[1]
    return torch.stack((mx*2/(mw-1)-1,my*2/(mh-1)-1),dim=-1)[None]


def _masked_ncc_loss(ref,mov,mask):
    den=mask.sum(dim=(-2,-1),keepdim=True).clamp_min(1.)
    ar=ref-(ref*mask).sum(dim=(-2,-1),keepdim=True)/den
    am=mov-(mov*mask).sum(dim=(-2,-1),keepdim=True)/den
    cross=(ar*am*mask).sum(dim=(-2,-1))
    mag=((ar.square()*mask).sum(dim=(-2,-1)) *
         (am.square()*mask).sum(dim=(-2,-1))).clamp_min(1e-7).sqrt()
    return 1-(cross/mag).mean()


def field_jacobian(flow, hw):
    """det(I + derivative of reference displacement); edge derivative excluded."""
    h,w=hw
    dx=(w-1)/(flow.shape[-1]-1)
    dy=(h-1)/(flow.shape[-2]-1)
    ux,uy=flow[:,0],flow[:,1]
    ux_x=(ux[:,:,2:]-ux[:,:,:-2])/(2*dx)
    uy_x=(uy[:,:,2:]-uy[:,:,:-2])/(2*dx)
    ux_y=(ux[:,2:,:]-ux[:,:-2,:])/(2*dy)
    uy_y=(uy[:,2:,:]-uy[:,:-2,:])/(2*dy)
    # crop to central region
    return (1+ux_x[:,1:-1,:])*(1+uy_y[:,:,1:-1])-ux_y[:,:,1:-1]*uy_x[:,1:-1,:]


def optimize_deformation(he,dapi,affine,levels=(256,512),
                         control_shapes=((9,23),(13,35)),iterations=(110,110),
                         max_displacement=70.,smoothness_weight=0.10,
                         magnitude_weight=0.006,fold_weight=3.,lr=.035,
                         device='cpu',verbose=True,feature_mode='multichannel',af_channel='auto'):
    """Fit low-dimensional smooth free-form field with image-only feature loss.

    The control field is in native H&E pixels. To reduce spurious changes on
    serial sections, start with small deformation and strong regularization.
    """
    assert len(levels)==len(control_shapes)==len(iterations)
    h,w=he.shape[:2]
    mh,mw=dapi.shape[:2]
    pyr=reg.feature_pyramid(he,dapi,levels=levels,mode=feature_mode,af_channel=af_channel,device=device)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    previous=None
    history=[]
    for stage,(lev,cshape,nsteps) in enumerate(zip(pyr,control_shapes,iterations)):
        output_hw=lev['ref'].shape[-2:]
        if previous is None:
            init=torch.zeros((1,2,*cshape),dtype=torch.float32,device=device)
        else:
            init=F.interpolate(previous,size=cshape,mode='bicubic',align_corners=True).detach()
        ctrl=torch.nn.Parameter(init)
        opt=torch.optim.Adam([ctrl],lr=lr*max_displacement)
        for it in range(nsteps):
            opt.zero_grad()
            # bounded field prevents pathological warps and exploding offsets
            bounded=max_displacement*torch.tanh(ctrl/max_displacement)
            flow=_upsample_flow(bounded,output_hw)
            grid=_sample_grid(output_hw,(h,w),(mh,mw),affine,flow)
            warped=F.grid_sample(lev['mov'],grid,align_corners=True,padding_mode='zeros')
            warped_mask=F.grid_sample(lev['mov_mask'],grid,align_corners=True,padding_mode='zeros')
            image_valid=((grid[...,0].abs()<1)&(grid[...,1].abs()<1)).float()[:,None]
            mask=lev['ref_mask']*warped_mask*image_valid
            loss_match=_masked_ncc_loss(lev['ref'],warped,mask)
            # derivative measured per physical H&E pixel
            ddx=(flow[:,:,:,1:]-flow[:,:,:,:-1])*(output_hw[1]-1)/(w-1)
            ddy=(flow[:,:,1:,:]-flow[:,:,:-1,:])*(output_hw[0]-1)/(h-1)
            smooth=(ddx.square().mean()+ddy.square().mean())
            # native pixel squared normalized by budget to be scale independent
            magnitude=(bounded/max_displacement).square().mean()
            jac=field_jacobian(flow,(h,w))
            fold=F.relu(.30-jac).square().mean()
            # discourage displacement of anatomy with no shared tissue support
            cov=mask.sum()/lev['ref_mask'].sum().clamp_min(1)
            coverage=3*F.relu(.85-cov).square()
            total=(loss_match+smoothness_weight*smooth+
                   magnitude_weight*magnitude+fold_weight*fold+coverage)
            total.backward()
            opt.step()
            history.append((stage,it,float(total.detach()),float(1-loss_match.detach()),
                            float(smooth.detach()),float(magnitude.detach()),
                            float(jac.min().detach()),float(cov.detach())))
        previous=(max_displacement*torch.tanh(ctrl/max_displacement)).detach()
        if verbose:
            print(f'Stage {stage+1}/{len(pyr)} @ {output_hw}: '
                  f'NCC={history[-1][3]:.4f}, minJac={history[-1][6]:.3f}, coverage={history[-1][7]:.3f}')
    return dict(controls=previous.cpu().numpy()[0],history=np.asarray(history),
                control_shape=previous.shape[-2:],reference_hw=(h,w),moving_hw=(mh,mw),
                affine_matrix=np.asarray(affine),max_displacement=max_displacement,
                configuration=dict(levels=list(levels),control_shapes=[list(t) for t in control_shapes],
                    iterations=list(iterations),smoothness_weight=smoothness_weight,
                    magnitude_weight=magnitude_weight,fold_weight=fold_weight))


def dense_field(result,hw):
    c=np.asarray(result['controls'],np.float32)
    # cv2 resizing multi-channel gives [h,w,2]
    return cv2.resize(np.moveaxis(c,0,-1),(hw[1],hw[0]),interpolation=cv2.INTER_CUBIC)


def warp_native_multichannel(dapi,ref_hw,result,block_rows=300):
    """Memory-friendly native uint16 remapping; exact AF intensities preserved."""
    h,w=ref_hw; A=result['affine_matrix']
    out=np.zeros((h,w)+dapi.shape[2:],dtype=dapi.dtype)
    ctrl=np.moveaxis(np.asarray(result['controls'],np.float32),0,-1)
    # use horizontal full-size control-upsample for consistency across blocks
    # full flow ~60MB for 4320x1700, safe for this input
    full=cv2.resize(ctrl,(w,h),interpolation=cv2.INTER_CUBIC)
    for y0 in range(0,h,block_rows):
        y1=min(h,y0+block_rows)
        yy,xx=np.mgrid[y0:y1,0:w].astype(np.float32)
        px=xx+full[y0:y1,:,0];py=yy+full[y0:y1,:,1]
        mapx=A[0,0]*px+A[0,1]*py+A[0,2]
        mapy=A[1,0]*px+A[1,1]*py+A[1,2]
        if dapi.ndim==3 and dapi.shape[-1]>4:
            for channel in range(dapi.shape[-1]):
                out[y0:y1,:,channel]=cv2.remap(dapi[...,channel],mapx.astype('float32'),mapy.astype('float32'),
                              cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        else:
            out[y0:y1]=cv2.remap(dapi,mapx.astype('float32'),mapy.astype('float32'),
                              cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return out


def evaluate_registration(he,dapi,affine,controls=None,side=768):
    lev=reg.feature_pyramid(he,dapi,levels=(side,),mode='nuclear_dapi')[0]
    hw=he.shape[:2];mh=dapi.shape[:2]
    flow=torch.zeros((1,2,*lev['ref'].shape[-2:]))
    if controls is not None:
        ct=torch.tensor(controls[None],dtype=torch.float32)
        flow=_upsample_flow(ct,lev['ref'].shape[-2:])
    with torch.no_grad():
        grid=_sample_grid(lev['ref'].shape[-2:],hw,mh,affine,flow)
        mov=F.grid_sample(lev['mov'],grid,align_corners=True)
        mask=lev['ref_mask']*F.grid_sample(lev['mov_mask'],grid,align_corners=True)
        ncc=float(1-_masked_ncc_loss(lev['ref'],mov,mask))
        coverage=float(mask.sum()/lev['ref_mask'].sum().clamp_min(1))
        jac=field_jacobian(flow,hw)
    return dict(nuclear_ncc=ncc,coverage=coverage,min_jacobian=float(jac.min()),
                folding_fraction=float((jac<=0).float().mean()))


def save_deformable(path,result,registered):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    tifffile.imwrite(path/'dapi_affine_deformable_registered.tif',registered)
    np.save(path/'deformable_control_field.npy',result['controls'])
    np.save(path/'affine_he_to_dapi.npy',result['affine_matrix'])
    np.save(path/'deformation_history.npy',result['history'])
    return path

# Backwards compatibility for original H&E/DAPI notebooks
warp_native_dapi = warp_native_multichannel
