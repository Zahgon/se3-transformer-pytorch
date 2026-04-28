from math import pi, sqrt
from functools import reduce
from operator import mul
import torch

from functools import lru_cache
from se3_transformer_pytorch.utils import cache

# constants

CACHE = {}

def clear_spherical_harmonics_cache():
    pass

def lpmv_cache_key_fn(l, m, x):
    pass

# spherical harmonics

@lru_cache(maxsize = 1000)
def semifactorial(x):
    pass

@lru_cache(maxsize = 1000)
def pochhammer(x, k):
    pass

def negative_lpmv(l, m, y):
    pass

@cache(cache = CACHE, key_fn = lpmv_cache_key_fn)
def lpmv(l, m, x):
    """Associated Legendre function including Condon-Shortley phase.

    Args:
        m: int order 
        l: int degree
        x: float argument tensor
    Returns:
        tensor of x-shape
    """
    pass

def get_spherical_harmonics_element(l, m, theta, phi):
    """Tesseral spherical harmonic with Condon-Shortley phase.

    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.

    Args:
        l: int for degree
        m: int for order, where -l <= m < l
        theta: collatitude or polar angle
        phi: longitude or azimuth
    Returns:
        tensor of shape theta
    """
    pass

def get_spherical_harmonics(l, theta, phi):
    """ Tesseral harmonic with Condon-Shortley phase.

    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.

    Args:
        l: int for degree
        theta: collatitude or polar angle
        phi: longitude or azimuth
    Returns:
        tensor of shape [*theta.shape, 2*l+1]
    """
    pass
