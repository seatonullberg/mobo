import numpy as np
from scipy.stats import gaussian_kde


class BaseSampler(object):
    """Representation of a general distribution sampler."""

    def __init__(self):
        pass

    def from_prior(self, *args, **kwargs):
        err = ("{} does not implement the required `from_prior` method."
               .format(self.__class__.__name__))
        raise NotImplementedError(err)

    def draw(self, *args, **kwargs):
        err = ("{} does not implement the required `draw` method."
               .format(self.__class__.__name__))
        raise NotImplementedError(err)


class GaussianSampler(BaseSampler):
    """Implementation of the Gaussian distribution sampler.
    
    Args:
        mean (optional) (numpy.ndarray): Mean of the distribution.
        std (optional) (numpy.ndarray): Standard deviation of the distribution.
    """

    def __init__(self, mean=None, std=None):
        super().__init__()
        assert all((type(mean) in [np.ndarray, type(None)],
                    type(std) in [np.ndarray, type(None)]))
        if type(mean) is np.ndarray and type(std) is np.ndarray:
            assert mean.shape == std.shape
        self._mean = mean
        self._std = std

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def from_prior(self, arr):
        """Sets the required properties from an existing distribution.
        
        Args:
            arr (numpy.ndarray): Distribution used to determine properties.
        """
        super().__init__()
        assert type(arr) is np.ndarray
        self._mean = np.mean(arr, axis=0)
        self._std = np.std(arr, axis=0)

    def draw(self, n):
        """Draws samples from a Gaussian distribution.
        
        Args:
            n (int): Number of samples to draw.
        
        Returns:
            numpy.ndarray
        """
        assert type(n) is int
        if self.mean is None or self.std is None:
            err = "`self.mean` and `self.std` must be set before sampling."
            raise ValueError(err)
        return np.random.normal(loc=self.mean,
                                scale=self.std,
                                size=(n, self.mean.shape[0]))
        

class KDESampler(BaseSampler):
    """Implementation of the Kernel Density Estimation sampler.
    
    Args:
        arr (optional) (numpy.ndarray): Array of costs to apply KDE to.
        bw (optional) (float): KDE bandwidth parameter.
    """

    def __init__(self, arr=None, bw=None):
        super().__init__()
        assert type(arr) in [np.ndarray, type(None)]
        assert type(bw) in [float, type(None)]
        self._arr = arr
        self._bw = bw

    @property
    def arr(self):
        return self._arr

    @property
    def bw(self):
        return self._bw

    def from_prior(self, arr):
        """Sets the required properties from an existing distribution.
        
        Args:
            arr (numpy.ndarray): Distribution used to determine properties.
        """
        assert type(arr) is np.ndarray
        kde = gaussian_kde(dataset=arr.T, bw_method="scott")
        self._arr = arr
        self._bw = kde.factor

    def draw(self, n):
        """Draws samples from a KDE distribution.
        
        Args:
            n (int): Number of samples to draw.

        Returns:
            numpy.ndarray
        """
        # TODO: figure out why transpose is needed
        assert type(n) is int
        if self.arr is None or self.bw is None:
            err = "`self.arr` and `self.bw` must be set before sampling."
            raise ValueError(err)
        kde = gaussian_kde(dataset=self.arr.T,
                           bw_method=self.bw)
        return kde.resample(n).T

class UniformSampler(BaseSampler):
    """Implementation of the uniform distribution sampler.
    
    Args:
        low (optional) (numpy.ndarray): Lower bound of the sampling interval.
        high (optional) (numpy.ndarray): Upper bound of the sampling interval.
    """

    def __init__(self, low=None, high=None):
        super().__init__()
        assert all((type(low) in [np.ndarray, type(None)],
                    type(high) in [np.ndarray, type(None)]))
        if type(low) is np.ndarray and type(high) is np.ndarray:
            assert low.shape == high.shape
        self._low = low
        self._high = high

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    def from_prior(self, arr):
        """Sets the required properties from an existing distribution.
        
        Args:
            arr (numpy.ndarray): Distribution used to determine properties.
        """
        assert type(arr) is np.ndarray
        self._low = arr.min(axis=0)
        self._high = arr.max(axis=0)

    def draw(self, n):
        """Draws samples from a uniform distribution.
        
        Args:
            n (int): Number of samples to draw.

        Returns:
            numpy.ndarray
        """
        assert type(n) is int
        if self.low is None or self.high is None:
            err = "`self.low` and `self.high` must be set before sampling."
        return np.random.uniform(low=self.low,
                                 high=self.high,
                                 size=(n, self.low.shape[0]))
