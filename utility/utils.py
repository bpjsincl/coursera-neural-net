"""Utility functions for Assignments."""
import scipy.io as sio


__all__ = ['zip_safe',
           'loadmat']


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
