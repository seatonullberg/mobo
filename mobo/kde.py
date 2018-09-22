"""
Functions for KDE bandwidth estimation
"""

import numpy as np
from scipy import stats
from scipy import optimize


def silverman_h(arr):
    '''
    Silverman smoothing factor
    :param arr: array
    :return: bandwidth
    '''
    kde = stats.gaussian_kde(arr, 'silverman')
    return kde.factor


def chiu_h(arr):
    '''
    Chiu cross validation method
    https://projecteuclid.org/download/pdf_1/euclid.aos/1176348376
    :param arr: array
    :return: bandwidth
    '''
    cb = ChiuBandwidth(arr)
    return cb.bandwidth


class ChiuBandwidth(object):

    def __init__(self, arr):
        self.arr = arr

    @property
    def bandwidth(self):
        h0 = .5
        results = optimize.minimize(lambda h: self.J(h),
                                    h0,
                                    method='Nelder-Mead')
        return results.x[0]

    def fhati(self, h, i):
        if type(h) is float:
            _h = h
        else:
            _h = h[0]

        arr_i = np.delete(self.arr, i)
        kde = stats.gaussian_kde(arr_i, _h)
        return kde.evaluate(self.arr[i])

    def J(self, h):
        if type(h) is float:
            _h = h
        else:
            _h = h[0]

        fhat = stats.gaussian_kde(self.arr, _h)
        F1 = fhat.integrate_kde(fhat)
        F2 = np.array([self.fhati(h, i) for i in range(self.arr.shape[0])])
        return F1 - 2 * np.mean(F2)
