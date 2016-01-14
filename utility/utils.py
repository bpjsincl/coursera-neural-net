"""Utility functions for Assignments."""
import scipy.io as sio
import numpy as np


__all__ = ['zip_safe',
           'loadmat',
           'logistic',
           'log_sum_exp_over_rows',
           'batches']


def zip_safe(*lists):
    """Zip function that checks that all the zipped lists have the same length."""
    assert len(lists) > 0
    assert all(len(list_) == len(lists[0]) for list_ in lists)
    zipped_lists = zip(*lists)
    return zipped_lists


def loadmat(filename):
    """This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.
    """
    return _check_keys(sio.loadmat(filename, struct_as_record=False, squeeze_me=True))


def _check_keys(data):
    """Checks if entries in dictionary are mat-objects.
    If yes todict is called to change them to nested dictionaries
    """
    for key in data:
        if isinstance(data[key], sio.matlab.mio5_params.mat_struct):
            data[key] = _todict(data[key])
    return data


def _todict(matobj):
    """A recursive function which constructs from matobjects nested dictionaries.
    """
    data = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            data[strg] = _todict(elem)
        else:
            data[strg] = elem
    return data


def logistic(x):
    return 1. / (1. + np.exp(-x))


def log_sum_exp_over_rows(a):
    """Computes log(sum(np.exp(a), 1)) in a numerically stable way."""
    col_maxs = np.max(a, axis=0)
    return np.log(sum(np.exp(a - np.tile(col_maxs, (np.size(a, 0), 1))), 0)) + col_maxs


def batches(iterable, n=1):
    """Yields specified number of mini batches (more efficient than extract_mini_batch(..)

    Args:
        iterable (numpy.array)  : helper function for splitting array into batches.
        n (int)                 : number of batches.
    """
    l = np.size(iterable, 1)
    for ndx in range(0, l, n):
        yield iterable[:, ndx:min(ndx + n, l)]
