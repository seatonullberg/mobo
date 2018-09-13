"""
Functions for calculating the pareto front
"""

import numpy as np


def calculate_pareto(arr):
    '''
    :param arr: (n_points, n_params) array
    :return: mask (n_pts,) boolean array mask of efficient points
    '''
    eff_pts = np.arange(arr.shape[0])
    n_pts = arr.shape[0]
    next_point_index = 0

    while next_point_index < len(arr):
        nondominated_point_mask = np.any(arr <= arr[next_point_index],
                                         axis=1)
        eff_pts = eff_pts[nondominated_point_mask]
        arr = arr[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1

    mask = np.zeros(n_pts, dtype=bool)
    mask[eff_pts] = True
    return mask
