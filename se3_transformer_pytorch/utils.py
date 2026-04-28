import os
import sys
import time
import pickle
import gzip
import torch
import contextlib
from functools import wraps, lru_cache
from filelock import FileLock

from einops import rearrange

# helper functions


def exists(val):
    pass


def default(val, d):
    pass


def uniq(arr):
    pass


def to_order(degree):
    pass


def map_values(fn, d):
    pass


def safe_cat(arr, el, dim):
    pass


def cast_tuple(val, depth):
    pass


def broadcat(tensors, dim=-1):
    pass


def batched_index_select(values, indices, dim=1):
    pass


def masked_mean(tensor, mask, dim=-1):
    pass


def rand_uniform(size, min_val, max_val):
    pass


def fast_split(arr, splits, dim=0):
    pass


def fourier_encode(x, num_encodings=4, include_self=True, flatten=True):
    pass


# default dtype context manager


@contextlib.contextmanager
def torch_default_dtype(dtype):
    pass


def cast_torch_tensor(fn):
    pass


# benchmark tool


def benchmark(fn):
    pass


# caching functions


def cache(cache, key_fn):
    pass


# cache in directory


def cache_dir(dirname, maxsize=128):
    """
    Cache a function with a directory

    :param dirname: the directory path
    :param maxsize: maximum size of the RAM cache (there is no limit for the directory cache)
    """
    pass
