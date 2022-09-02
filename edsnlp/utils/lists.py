from ctypes import Union
from typing import List


def flatten(l: List):
    """
    Flatten (if necessary) a list of sublists

    Parameters
    ----------
    l : List
        A list of items, or a list of lists

    Returns
    -------
    List
        A flatten list
    """
    if not l:
        return l
    l = [item if isinstance(item, list) else [item] for item in l]

    return [item for sublist in l for item in sublist]
