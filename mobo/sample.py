import numpy as np
import scipy as sp


class BaseSampler(object):
    """Representation of a general distribution sampler."""

    def __init__(self):
        pass

    def draw(self, *args, **kwargs):
        err = ("{} does not implement the required `draw` method."
               .format(self.__class__.__name__))
        raise NotImplementedError(err)


class GaussianSampler(BaseSampler):
    """Implementation of the Gaussian distribution sampler.
    
    Args:
        mean (numpy.ndarray): Mean of the distribution.
        stdev (numpy.ndarray): Standard deviation of the distribution.
    """

    def __init__(self, mean, stdev):
        super().__init__()
        assert all((type(mean) is np.ndarray,
                    type(stdev) is np.ndarray))
        assert mean.shape == stdev.shape
        self._mean = mean
        self._stdev = stdev

    @property
    def mean(self):
        return self._mean

    @property
    def stdev(self):
        return self._stdev

    def draw(self, n):
        """Draws samples from a Gaussian distribution.
        
        Args:
            n (int): Number of samples to draw.
        
        Returns:
            numpy.ndarray
        """
        assert type(n) is int
        return np.random.normal(loc=self.mean,
                                scale=self.stdev,
                                size=(n, self.mean.shape[0]))
        

class KDESampler(BaseSampler):
    """Implementation of the Kernel Density Estimation sampler.
    
    Args:
        arr (numpy.ndarray): Array of costs to apply KDE to.
        bw (float): KDE bandwidth parameter.
    """

    def __init__(self, arr, bw):
        super().__init__()
        assert type(arr) is np.ndarray
        assert type(bw) is float
        self._arr = arr
        self._bw = bw

    @property
    def arr(self):
        return self._arr

    @property
    def bw(self):
        return self._bw

    def draw(self, n):
        """Draws samples from a KDE distribution.
        
        Args:
            n (int): Number of samples to draw.

        Returns:
            numpy.ndarray
        """
        # TODO: figure out why transpose is needed
        assert type(n) is int
        kde = sp.stats.gaussian_kde(dataset=self.arr.T,
                                    bw_method=self.bw)
        return kde.resample(n).T

class UniformSampler(BaseSampler):
    """Implementation of the uniform distribution sampler.
    
    Args:
        low (numpy.ndarray): Lower bound of the sampling interval.
        high (numpy.ndarray): Upper bound of the sampling interval.
    """

    def __init__(self, low, high):
        super().__init__()
        assert all((type(low) is float,
                    type(high) is float))
        assert low.shape == high.shape
        self._low = low
        self._high = high

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    def draw(self, n):
        """Draws samples from a uniform distribution.
        
        Args:
            n (int): Number of samples to draw.

        Returns:
            numpy.ndarray
        """
        assert type(n) is int
        return np.random.uniform(low=self.low,
                                 high=self.high,
                                 size=(n, self.low.shape[0]))
