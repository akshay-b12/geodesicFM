"""Explicit domains, characteristic step scales and fixed auxiliary fields.

Registered scales are physical reference units, not data-fitted or relative to
the current parameter value. The same epsilon sweep is thus reproducible.
"""
from dataclasses import dataclass
import math

import torch

from .datasets import core_crop
from .degradations import (affine_intensity, gaussian_noise, gaussian_blur,
    anisotropic_blur, haze, blur_noise, blur_haze)
from .degradations.common import convolve


@dataclass
class OperatorSpec:
    name: str
    parameter_names: tuple
    scales: tuple
    bounds: tuple
    points: list
    has_noise: bool = False
    has_blur: bool = False


def operator_specs(sigma_domain):
    lo, hi = map(float,sigma_domain)
    mid = (lo+hi)/2
    blur_box = (lo,hi)
    common_blur = [lo,mid,hi]
    return {
        "affine_intensity": OperatorSpec("affine_intensity",("a","b"),(1.0,0.1),((0.0,2.0),(-0.5,0.5)),
                                         [[1.0,0.0],[1.5,-0.1]]),
        "gaussian_noise": OperatorSpec("gaussian_noise",("sigma_noise",),(0.1,),((0.0,0.2),),[[0.0],[0.05],[0.15]],True),
        "isotropic_blur": OperatorSpec("isotropic_blur",("sigma",),(1.0,),(blur_box,),[[s] for s in common_blur],has_blur=True),
        "anisotropic_fixed_theta": OperatorSpec("anisotropic_fixed_theta",("sigma_x","sigma_y"),(1.0,1.0),
                    (blur_box,blur_box),[[lo,mid],[mid,hi],[hi,lo]],has_blur=True),
        "anisotropic_blur": OperatorSpec("anisotropic_blur",("sigma_x","sigma_y","theta_rad"),(1.0,1.0,1.0),
                    (blur_box,blur_box,(0.0,math.pi)),
                    [[hi,lo,0.0],[hi,lo,0.63],[mid,mid,0.47],[min(hi,mid*1.001),mid,0.47]],has_blur=True),
        "haze_unit_depth": OperatorSpec("haze_unit_depth",("beta","A"),(1.0,1.0),((0.0,2.0),(0.0,1.0)),[[0.0,0.9],[0.5,0.9],[1.3,0.5]]),
        "haze_toy_depth": OperatorSpec("haze_toy_depth",("beta","A"),(1.0,1.0),((0.0,2.0),(0.0,1.0)),[[0.0,0.9],[0.5,0.9],[1.3,0.5]]),
        **{name:OperatorSpec(name,("sigma_blur","sigma_noise"),(1.0,0.1),
                            (blur_box,(0.0,0.2)),[[lo,0.0],[mid,0.05],[hi,0.1]],True,True)
           for name in ("blur_then_noise","noise_then_blur")},
        **{name:OperatorSpec(name,("sigma_blur","beta","A"),(1.0,1.0,1.0),
                            (blur_box,(0.0,2.0),(0.0,1.0)),[[lo,0.0,0.9],[mid,0.5,0.9],[hi,1.3,0.5]],has_blur=True)
           for name in ("blur_then_haze","haze_then_blur")},
    }


def auxiliary_fields(x, seed):
    """Master float64 draws may be cast to float32 for like-for-like comparisons."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    epsilon = torch.randn(x.shape,generator=generator,dtype=torch.float64).to(x.device,x.dtype)
    unit = torch.ones((1,*x.shape[-2:]),dtype=x.dtype,device=x.device)
    depth = torch.linspace(0.25,1.75,x.shape[-1],dtype=x.dtype,device=x.device)[None,None,:]
    depth = depth.expand(1,x.shape[-2],x.shape[-1]).contiguous()
    return {"epsilon":epsilon,"unit_depth":unit,"toy_depth":depth}


def bind_operator(spec, x, *, halo, core_size, kernel_size, fields):
    """Freeze image, depth, noise, support, composition order and crop context."""
    if spec.has_blur and halo < kernel_size//2:
        raise ValueError("Insufficient physical halo for measured blur core.")
    name = spec.name
    def context_fn(p):
        if name == "affine_intensity":
            return affine_intensity(x,p)
        if name == "gaussian_noise":
            return gaussian_noise(x,p,epsilon=fields["epsilon"])
        if name == "isotropic_blur":
            return gaussian_blur(x,p,kernel_size)
        if name == "anisotropic_fixed_theta":
            q = torch.stack((p[0],p[1],p.new_tensor(0.31)))
            return anisotropic_blur(x,q,kernel_size)
        if name == "anisotropic_blur":
            return anisotropic_blur(x,p,kernel_size)
        if name in ("haze_unit_depth","haze_toy_depth"):
            return haze(x,p,depth=fields["unit_depth" if name == "haze_unit_depth" else "toy_depth"])
        if name in ("blur_then_noise","noise_then_blur"):
            return blur_noise(x,p,epsilon=fields["epsilon"],kernel_size=kernel_size,order=name)
        if name in ("blur_then_haze","haze_then_blur"):
            return blur_haze(x,p,depth=fields["toy_depth"],kernel_size=kernel_size,order=name)
        raise ValueError(f"Unknown operator: {name}")
    return lambda p: core_crop(context_fn(p),halo,core_size)


def analytic_isotropic_derivative(x, sigma, kernel_size):
    """d normalized sampled Gaussian / d sigma, without automatic differentiation.

    For normalized weights k=w/sum(w), dk=k*(d log(w)-E_k[d log(w)]).
    Kernel-normalization derivatives are essential; omitting them changes J.
    """
    q = torch.arange(-(kernel_size//2),kernel_size//2+1,dtype=x.dtype,device=x.device)
    weight = torch.exp(-q.square()/(2*sigma.square()))
    k = weight/weight.sum()
    score = q.square()/sigma.pow(3)
    dk = k*(score-(k*score).sum())
    return (convolve(convolve(x,dk[None,:]),k[:,None])
            + convolve(convolve(x,k[None,:]),dk[:,None]))


def analytic_anisotropic_derivatives(x, p, kernel_size):
    """Closed-form score derivatives, including the normalized orientation term."""
    sx,sy,theta = p.unbind()
    q = torch.arange(-(kernel_size//2),kernel_size//2+1,dtype=x.dtype,device=x.device)
    yy,xx = torch.meshgrid(q,q,indexing="ij")
    u = theta.cos()*xx+theta.sin()*yy
    v = -theta.sin()*xx+theta.cos()*yy
    weight = torch.exp(-0.5*((u/sx).square()+(v/sy).square()))
    k = weight/weight.sum()
    scores = (u.square()/sx.pow(3),v.square()/sy.pow(3),u*v*(1/sy.square()-1/sx.square()))
    return torch.stack([convolve(x,k*(score-(k*score).sum())) for score in scores])


def analytic_jacobian(spec, x, p, *, halo, core_size, fields, kernel_size=33):
    """Closed-form and explicit chain-rule references, without AD/FD calls."""
    name = spec.name
    if name == "affine_intensity":
        return torch.stack((core_crop(x,halo,core_size),torch.ones_like(core_crop(x,halo,core_size))))
    if name == "gaussian_noise":
        return core_crop(fields["epsilon"],halo,core_size).unsqueeze(0)
    if name in ("haze_unit_depth","haze_toy_depth"):
        depth = fields["unit_depth" if name == "haze_unit_depth" else "toy_depth"]
        t = torch.exp(-p[0]*depth)
        dbeta = depth*t*(p[1]-x)
        dairlight = (1-t).expand_as(x)
        return torch.stack((core_crop(dbeta,halo,core_size),core_crop(dairlight,halo,core_size)))
    if name == "isotropic_blur":
        return core_crop(analytic_isotropic_derivative(x,p[0],kernel_size),halo,core_size).unsqueeze(0)
    if name in ("anisotropic_fixed_theta","anisotropic_blur"):
        q = torch.stack((p[0],p[1],p.new_tensor(0.31))) if name == "anisotropic_fixed_theta" else p
        reference = analytic_anisotropic_derivatives(x,q,kernel_size)
        return core_crop(reference[:2] if name == "anisotropic_fixed_theta" else reference,halo,core_size)
    if name in ("blur_then_noise","noise_then_blur"):
        eps = fields["epsilon"]
        if name == "blur_then_noise":
            derivatives = (analytic_isotropic_derivative(x,p[0],kernel_size),eps)
        else:
            derivatives = (analytic_isotropic_derivative(x+p[1]*eps,p[0],kernel_size),
                           gaussian_blur(eps,p[:1],kernel_size))
        return torch.stack([core_crop(v,halo,core_size) for v in derivatives])
    if name in ("blur_then_haze","haze_then_blur"):
        depth = fields["toy_depth"]
        t = torch.exp(-p[1]*depth)
        if name == "blur_then_haze":
            blurred = gaussian_blur(x,p[:1],kernel_size)
            derivatives = (t*analytic_isotropic_derivative(x,p[0],kernel_size),
                           depth*t*(p[2]-blurred),(1-t).expand_as(x))
        else:
            hazed = x*t+p[2]*(1-t)
            derivatives = (analytic_isotropic_derivative(hazed,p[0],kernel_size),
                           gaussian_blur(depth*t*(p[2]-x),p[:1],kernel_size),
                           gaussian_blur((1-t).expand_as(x),p[:1],kernel_size))
        return torch.stack([core_crop(v,halo,core_size) for v in derivatives])
    return None


def synthetic_context(pattern, core_size, halo, seed=57721):
    side = core_size+2*halo
    q = torch.linspace(0,1,side,dtype=torch.float64)
    yy,xx = torch.meshgrid(q,q,indexing="ij")
    if pattern == "constant":
        return torch.ones((3,side,side),dtype=torch.float64)*0.37
    if pattern == "impulse":
        x = torch.zeros((3,side,side),dtype=torch.float64)
        x[:,side//2,side//2] = torch.tensor([1.0,0.7,0.4],dtype=x.dtype)
        return x
    if pattern == "textured":
        x = torch.stack((xx,yy,0.5+0.25*torch.sin(70*xx)*torch.cos(45*yy)))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return (x+0.04*torch.randn(x.shape,generator=generator,dtype=x.dtype)).clamp(0,1)
    raise ValueError("Unknown control pattern.")
