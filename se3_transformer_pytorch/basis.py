import os
from math import pi
import torch
from torch import einsum
from einops import rearrange
from itertools import product
from contextlib import contextmanager

from se3_transformer_pytorch.irr_repr import irr_repr, spherical_harmonics
from se3_transformer_pytorch.utils import torch_default_dtype, cache_dir, exists, default, to_order
from se3_transformer_pytorch.spherical_harmonics import clear_spherical_harmonics_cache

# constants

CACHE_PATH = default(os.getenv('CACHE_PATH'), os.path.expanduser('~/.cache.equivariant_attention'))
CACHE_PATH = CACHE_PATH if not exists(os.environ.get('CLEAR_CACHE')) else None

# todo (figure ot why this was hard coded in official repo)

RANDOM_ANGLES = [ 
    [4.41301023, 5.56684102, 4.59384642],
    [4.93325116, 6.12697327, 4.14574096],
    [0.53878964, 4.09050444, 5.36539036],
    [2.16017393, 3.48835314, 5.55174441],
    [2.52385107, 0.2908958, 3.90040975]
]

# helpers

@contextmanager
def null_context():
    pass

# functions

def get_matrix_kernel(A, eps = 1e-10):
    '''
    Compute an orthonormal basis of the kernel (x_1, x_2, ...)
    A x_i = 0
    scalar_product(x_i, x_j) = delta_ij

    :param A: matrix
    :return: matrix where each row is a basis vector of the kernel of A
    '''
    pass


def get_matrices_kernel(As, eps = 1e-10):
    '''
    Computes the common kernel of all the As matrices
    '''
    pass

def get_spherical_from_cartesian(cartesian, divide_radius_by = 1.0):
    """
    # ON ANGLE CONVENTION
    #
    # sh has following convention for angles:
    # :param theta: the colatitude / polar angle, ranging from 0(North Pole, (X, Y, Z) = (0, 0, 1)) to pi(South Pole, (X, Y, Z) = (0, 0, -1)).
    # :param phi: the longitude / azimuthal angle, ranging from 0 to 2 pi.
    #
    # the 3D steerable CNN code therefore (probably) has the following convention for alpha and beta:
    # beta = pi - theta; ranging from 0(South Pole, (X, Y, Z) = (0, 0, -1)) to pi(North Pole, (X, Y, Z) = (0, 0, 1)).
    # alpha = phi
    #
    """
    pass

def kron(a, b):
    """
    A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk

    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    pass

def get_R_tensor(order_out, order_in, a, b, c):
    pass

def sylvester_submatrix(order_out, order_in, J, a, b, c):
    ''' generate Kronecker product matrix for solving the Sylvester equation in subspace J '''
    pass

@cache_dir(CACHE_PATH)
@torch_default_dtype(torch.float64)
@torch.no_grad()
def basis_transformation_Q_J(J, order_in, order_out, random_angles = RANDOM_ANGLES):
    """
    :param J: order of the spherical harmonics
    :param order_in: order of the input representation
    :param order_out: order of the output representation
    :return: one part of the Q^-1 matrix of the article
    """
    pass

def precompute_sh(r_ij, max_J):
    """
    pre-comput spherical harmonics up to order max_J

    :param r_ij: relative positions
    :param max_J: maximum order used in entire network
    :return: dict where each entry has shape [B,N,K,2J+1]
    """
    pass

def get_basis(r_ij, max_degree, differentiable = False):
    """Return equivariant weight basis (basis)

    Call this function *once* at the start of each forward pass of the model.
    It computes the equivariant weight basis, W_J^lk(x), and internodal 
    distances, needed to compute varphi_J^lk(x), of eqn 8 of
    https://arxiv.org/pdf/2006.10503.pdf. The return values of this function 
    can be shared as input across all SE(3)-Transformer layers in a model.

    Args:
        r_ij: relative positional vectors
        max_degree: non-negative int for degree of highest feature-type
        differentiable: whether r_ij should receive gradients from basis
    Returns:
        dict of equivariant bases, keys are in form '<d_in><d_out>'
    """
    pass
