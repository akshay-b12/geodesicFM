"""Flat control D_(a,b)(x)=a*x+b, deliberately without clipping."""
from .common import parameters


def affine_intensity(x, xi):
    a, b = parameters(x, xi, 2).unbind()
    return a * x + b
