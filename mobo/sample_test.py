from mobo.sample import GaussianSampler, KDESampler, UniformSampler
import numpy as np


def test_gaussian_sampler():
    n = 10
    mean = np.array([0.0, 0.0, 0.0])
    stdev = np.array([1.0, 1.0, 1.0])
    gs = GaussianSampler(mean=mean, stdev=stdev)
    samples = gs.draw(n)
    assert type(samples) is np.ndarray
    assert samples.shape[0] == 10
    assert samples.shape[1] == 3


def test_kde_sampler():
    n = 10
    arr = np.random.normal(size=(100, 3))
    bw = 0.1
    ks = KDESampler(arr=arr, bw=bw)
    samples = ks.draw(n)
    assert type(samples) is np.ndarray
    assert samples.shape[0] == 10
    assert samples.shape[1] == 3


def test_uniform_sampler():
    n = 10
    low = np.array([0.0, 0.0, 0.0])
    high = np.array([1.0, 1.0, 1.0])
    us = UniformSampler(low=low, high=high)
    samples = us.draw(n)
    assert type(samples) is np.ndarray
    assert samples.shape[0] == 10
    assert samples.shape[1] == 3
    assert samples.min() >= 0
    assert samples.max() <= 1
