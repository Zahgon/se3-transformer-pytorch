import os
import numpy as np
import torch
from torch import sin, cos, atan2, acos
from math import pi
from pathlib import Path
from functools import wraps

from se3_transformer_pytorch.utils import exists, default, cast_torch_tensor, to_order
from se3_transformer_pytorch.spherical_harmonics import get_spherical_harmonics, clear_spherical_harmonics_cache

DATA_PATH = path = Path(os.path.dirname(__file__)) / 'data'

try:
    path = DATA_PATH / 'J_dense.pt'
    Jd = torch.load(str(path))
except:
    path = DATA_PATH / 'J_dense.npy'
    Jd_np = np.load(str(path), allow_pickle = True)
    Jd = list(map(torch.from_numpy, Jd_np))

def wigner_d_matrix(degree, alpha, beta, gamma, dtype = None, device = None):
    """Create wigner D matrices for batch of ZYZ Euler anglers for degree l."""
    pass

def z_rot_mat(angle, l):
    pass

def irr_repr(order, alpha, beta, gamma, dtype = None):
    """
    irreducible representation of SO3
    - compatible with compose and spherical_harmonics
    """
    pass

@cast_torch_tensor
def rot_z(gamma):
    '''
    Rotation around Z axis
    '''
    pass

@cast_torch_tensor
def rot_y(beta):
    '''
    Rotation around Y axis
    '''
    pass

@cast_torch_tensor
def x_to_alpha_beta(x):
    '''
    Convert point (x, y, z) on the sphere into (alpha, beta)
    '''
    pass

def rot(alpha, beta, gamma):
    '''
    ZYZ Euler angles rotation
    '''
    pass

def compose(a1, b1, c1, a2, b2, c2):
    """
    (a, b, c) = (a1, b1, c1) composed with (a2, b2, c2)
    """
    pass

def spherical_harmonics(order, alpha, beta, dtype = None):
    pass
