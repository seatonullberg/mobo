import numpy as np
from scipy.stats import gaussian_kde
from typing import Any


class BaseSampler(object):
    """Abstract base class for Samplers."""
    @classmethod
    def from_prior(cls, distribution: np.ndarray):
        raise NotImplementedError()

    def draw(self, n_samples: int):
        raise NotImplementedError()


class GaussianSampler(BaseSampler):
    """Gaussian distribution sampler.
    
    Args:
        mean: Mean of the distribution.
        std: Standard deviation of the distribution.
    """
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean
        self.std = std

    @classmethod
    def from_prior(cls, distribution: np.ndarray):
        """Initializes the sampler from a prior distribution.
        
        Args:
            distribution: Array representing the prior distribution.
        """
        mean = np.mean(distribution, axis=0)
        std = np.std(distribution, axis=0)
        return cls(mean, std)

    def draw(self, n_samples: int) -> np.ndarray:
        """Draws samples from the distribution.
        
        Args:
            n_samples: Number of samples to draw.
        """
        return np.random.normal(loc=self.mean, 
                                scale=self.std, 
                                size=(n_samples, self.mean.shape[0]))


class KDESampler(BaseSampler):
    """Kernel Density Estimation sampler.
    
    Args:
        bandwidth_method: Bandwidth algorithm or value to use in estimation.
        distribution: A distribution to evaluate the KDE over.
    """
    def __init__(self, bandwidth_method: Any, distribution: np.ndarray) -> None:
        self.kde = gaussian_kde(distribution, bandwidth_method)

    @classmethod
    def from_prior(cls, distribution: gaussian_kde):
        return cls(distribution.factor, distribution.dataset)

    def draw(self, n_samples: int) -> np.ndarray:
        """Draws samples from the distribution.
        
        Args:
            n_samples: Number of samples to draw.
        """
        return self.kde.resample(n_samples).T


class UniformSampler(BaseSampler):
    """Uniform distribution sampler.
    
    Args:
        lower_bound: Lower bound of the sampling interval.
        upper_bound: Upper bound of the sampling interval.
    """
    def __init__(self, lower_bound: np.ndarray, 
                 upper_bound: np.ndarray) -> None:
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @classmethod
    def from_prior(cls, distribution: np.ndarray):
        return cls(distribution.min(axis=0),
                   distribution.max(axis=0))

    def draw(self, n_samples: int) -> np.ndarray:
        """Draws samples from the distribution.
        
        Args:
            n_samples: Number of samples to draw.
        """
        return np.random.uniform(low=self.lower_bound,
                                 high=self.upper_bound,
                                 size=(n_samples, self.lower_bound.shape[0]))
