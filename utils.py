"""Utility functions for Assignments."""

__all__ = ['zip_safe']


def zip_safe(*lists):
    """Zip function that checks that all the zipped lists have the same length."""
    assert len(lists) > 0
    assert all(len(list_) == len(lists[0]) for list_ in lists)
    zipped_lists = zip(*lists)
    return zipped_lists
