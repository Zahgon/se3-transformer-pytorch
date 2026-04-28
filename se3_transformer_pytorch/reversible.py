import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

# helpers


def map_values(fn, x):
    pass


def dict_chunk(x, chunks, dim):
    pass


def dict_sum(x, y):
    pass


def dict_subtract(x, y):
    pass


def dict_cat(x, y, dim):
    pass


def dict_set_(x, key, value):
    pass


def dict_backwards_(outputs, grad_tensors):
    pass


def dict_del_(x):
    pass


def values(d):
    pass


# following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
class Deterministic(nn.Module):
    def __init__(self, net):
        pass

    def record_rng(self, *args):
        pass

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        pass


# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source
class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        pass

    def forward(self, x, **kwargs):
        pass

    def backward_pass(self, y, dy, **kwargs):
        pass


class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        pass

    @staticmethod
    def backward(ctx, dy):
        pass


class SequentialSequence(nn.Module):
    def __init__(self, blocks):
        pass

    def forward(self, x, **kwargs):
        pass


class ReversibleSequence(nn.Module):
    def __init__(self, blocks):
        pass

    def forward(self, x, **kwargs):
        pass
