"""Explicit operator order; compositions do not silently commute."""
import torch
from .common import parameters
from .gaussian_blur import gaussian_blur
from .gaussian_noise import gaussian_noise
from .haze import haze


def blur_noise(x, xi, *, epsilon, kernel_size=33, order="blur_then_noise"):
    blur_sigma, noise_sigma = parameters(x, xi, 2).unbind()
    if order == "blur_then_noise":
        return gaussian_noise(gaussian_blur(x, blur_sigma, kernel_size), noise_sigma, epsilon=epsilon)
    if order == "noise_then_blur":
        return gaussian_blur(gaussian_noise(x, noise_sigma, epsilon=epsilon), blur_sigma, kernel_size)
    raise ValueError("order must be blur_then_noise or noise_then_blur")


def blur_haze(x, xi, *, depth, kernel_size=33, order="blur_then_haze"):
    blur_sigma, beta, airlight = parameters(x, xi, 3).unbind()
    hparams = torch.stack((beta, airlight))
    if order == "blur_then_haze":
        return haze(gaussian_blur(x, blur_sigma, kernel_size), hparams, depth=depth)
    if order == "haze_then_blur":
        return gaussian_blur(haze(x, hparams, depth=depth), blur_sigma, kernel_size)
    raise ValueError("order must be blur_then_haze or haze_then_blur")
