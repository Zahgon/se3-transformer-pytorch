from math import sqrt
from itertools import product
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, einsum

from se3_transformer_pytorch.basis import get_basis
from se3_transformer_pytorch.utils import (
    exists,
    default,
    uniq,
    map_values,
    batched_index_select,
    masked_mean,
    to_order,
    fourier_encode,
    cast_tuple,
    safe_cat,
    fast_split,
    rand_uniform,
    broadcat,
)
from se3_transformer_pytorch.reversible import ReversibleSequence, SequentialSequence
from se3_transformer_pytorch.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb

from einops import rearrange, repeat

# fiber helpers

FiberEl = namedtuple("FiberEl", ["degrees", "dim"])


class Fiber(nn.Module):
    def __init__(self, structure):
        pass

    @property
    def dims(self):
        pass

    @property
    def degrees(self):
        pass

    @staticmethod
    def create(num_degrees, dim):
        pass

    def __getitem__(self, degree):
        pass

    def __iter__(self):
        pass

    def __mul__(self, fiber):
        pass

    def __and__(self, fiber):
        pass


def get_tensor_device_and_dtype(features):
    pass


# classes


class ResidualSE3(nn.Module):
    """only support instance where both Fibers are identical"""

    def forward(self, x, res):
        pass


class LinearSE3(nn.Module):
    def __init__(self, fiber_in, fiber_out):
        pass

    def forward(self, x):
        pass


class NormSE3(nn.Module):
    """Norm-based SE(3)-equivariant nonlinearity.

    Nonlinearities are important in SE(3) equivariant GCNs. They are also quite
    expensive to compute, so it is convenient for them to share resources with
    other layers, such as normalization. The general workflow is as follows:

    > for feature type in features:
    >    norm, phase <- feature
    >    output = fnc(norm) * phase

    where fnc: {R+}^m -> R^m is a learnable map from m norms to m scalars.
    """

    def __init__(
        self,
        fiber,
        nonlin=nn.GELU(),
        gated_scale=False,
        eps=1e-12,
    ):
        pass

    def forward(self, features):
        pass


class ConvSE3(nn.Module):
    """A tensor field network layer

    ConvSE3 stands for a Convolution SE(3)-equivariant layer. It is the
    equivalent of a linear layer in an MLP, a conv layer in a CNN, or a graph
    conv layer in a GCN.

    At each node, the activations are split into different "feature types",
    indexed by the SE(3) representation type: non-negative integers 0, 1, 2, ..
    """

    def __init__(
        self,
        fiber_in,
        fiber_out,
        self_interaction=True,
        pool=True,
        edge_dim=0,
        fourier_encode_dist=False,
        num_fourier_features=4,
        splits=4,
    ):
        pass

    def forward(self, inp, edge_info, rel_dist=None, basis=None):
        pass


class RadialFunc(nn.Module):
    """NN parameterized radial profile function."""

    def __init__(self, num_freq, in_dim, out_dim, edge_dim=None, mid_dim=128):
        pass

    def forward(self, x):
        pass


class PairwiseConv(nn.Module):
    """SE(3)-equivariant convolution between two single-type features"""

    def __init__(self, degree_in, nc_in, degree_out, nc_out, edge_dim=0, splits=4):
        pass

    def forward(self, feat, basis):
        pass


# feed forwards


class FeedForwardSE3(nn.Module):
    def __init__(self, fiber, mult=4):
        pass

    def forward(self, features):
        pass


class FeedForwardBlockSE3(nn.Module):
    def __init__(self, fiber, norm_gated_scale=False):
        pass

    def forward(self, features):
        pass


# attention


class AttentionSE3(nn.Module):
    def __init__(
        self,
        fiber,
        dim_head=64,
        heads=8,
        attend_self=False,
        edge_dim=None,
        fourier_encode_dist=False,
        rel_dist_num_fourier_features=4,
        use_null_kv=False,
        splits=4,
        global_feats_dim=None,
        linear_proj_keys=False,
        tie_key_values=False,
    ):
        pass

    def forward(
        self,
        features,
        edge_info,
        rel_dist,
        basis,
        global_feats=None,
        pos_emb=None,
        mask=None,
    ):
        pass


# AttentionSE3, but with one key / value projection shared across all query heads
class OneHeadedKVAttentionSE3(nn.Module):
    def __init__(
        self,
        fiber,
        dim_head=64,
        heads=8,
        attend_self=False,
        edge_dim=None,
        fourier_encode_dist=False,
        rel_dist_num_fourier_features=4,
        use_null_kv=False,
        splits=4,
        global_feats_dim=None,
        linear_proj_keys=False,
        tie_key_values=False,
    ):
        pass

    def forward(
        self,
        features,
        edge_info,
        rel_dist,
        basis,
        global_feats=None,
        pos_emb=None,
        mask=None,
    ):
        pass


class AttentionBlockSE3(nn.Module):
    def __init__(
        self,
        fiber,
        dim_head=24,
        heads=8,
        attend_self=False,
        edge_dim=None,
        use_null_kv=False,
        fourier_encode_dist=False,
        rel_dist_num_fourier_features=4,
        splits=4,
        global_feats_dim=False,
        linear_proj_keys=False,
        tie_key_values=False,
        attention_klass=AttentionSE3,
        norm_gated_scale=False,
    ):
        pass

    def forward(
        self,
        features,
        edge_info,
        rel_dist,
        basis,
        global_feats=None,
        pos_emb=None,
        mask=None,
    ):
        pass


# egnn


class Swish_(nn.Module):
    def forward(self, x):
        pass


SiLU = nn.SiLU if hasattr(nn, "SiLU") else Swish_


class HtypesNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, scale_init=1e-2, bias_init=1e-2):
        pass

    def forward(self, coors):
        pass


class EGNN(nn.Module):
    def __init__(
        self,
        fiber,
        hidden_dim=32,
        edge_dim=0,
        init_eps=1e-3,
        coor_weights_clamp_value=None,
    ):
        pass

    def init_(self, module):
        pass

    def forward(self, features, edge_info, rel_dist, mask=None, **kwargs):
        pass


class EGnnNetwork(nn.Module):
    def __init__(
        self,
        *,
        fiber,
        depth,
        edge_dim=0,
        hidden_dim=32,
        coor_weights_clamp_value=None,
        feedforward=False,
    ):
        pass

    def forward(
        self,
        features,
        edge_info,
        rel_dist,
        basis,
        global_feats=None,
        pos_emb=None,
        mask=None,
        **kwargs,
    ):
        pass


# main class


class SE3Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads=8,
        dim_head=24,
        depth=2,
        input_degrees=1,
        num_degrees=None,
        output_degrees=1,
        valid_radius=1e5,
        reduce_dim_out=False,
        num_tokens=None,
        num_positions=None,
        num_edge_tokens=None,
        edge_dim=None,
        reversible=False,
        attend_self=True,
        use_null_kv=False,
        differentiable_coors=False,
        fourier_encode_dist=False,
        rel_dist_num_fourier_features=4,
        num_neighbors=float("inf"),
        attend_sparse_neighbors=False,
        num_adj_degrees=None,
        adj_dim=0,
        max_sparse_neighbors=float("inf"),
        dim_in=None,
        dim_out=None,
        norm_out=False,
        num_conv_layers=0,
        causal=False,
        splits=4,
        global_feats_dim=None,
        linear_proj_keys=False,
        one_headed_key_values=False,
        tie_key_values=False,
        rotary_position=False,
        rotary_rel_dist=False,
        norm_gated_scale=False,
        use_egnn=False,
        egnn_hidden_dim=32,
        egnn_weights_clamp_value=None,
        egnn_feedforward=False,
        hidden_fiber_dict=None,
        out_fiber_dict=None,
    ):
        pass

    def forward(
        self,
        feats,
        coors,
        mask=None,
        adj_mat=None,
        edges=None,
        return_type=None,
        return_pooled=False,
        neighbor_mask=None,
        global_feats=None,
    ):
        pass
