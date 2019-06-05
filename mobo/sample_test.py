from mobo.sample import GaussianSampler, KDESampler, UniformSampler
import numpy as np


n = 10
cols = 3
prior = np.random.uniform(size=(100, cols))


def test_gaussian_sampler():
    mean = np.array([0.0 for i in range(cols)])
    std = np.array([1.0 for i in range(cols)])
    gs = GaussianSampler(mean=mean, std=std)
    samples = gs.draw(n)
    assert type(samples) is np.ndarray
    assert samples.shape[0] == n
    assert samples.shape[1] == cols


def test_gaussian_sampler_from_prior():
    gs = GaussianSampler()
    gs.from_prior(prior)
    samples = gs.draw(n)
    assert type(samples) is np.ndarray
    assert samples.shape[0] == n
    assert samples.shape[1] == cols


def test_kde_sampler():
    arr = np.random.normal(size=(100, cols))
    bw = 0.1
    ks = KDESampler(arr=arr, bw=bw)
    samples = ks.draw(n)
    assert type(samples) is np.ndarray
    assert samples.shape[0] == n
    assert samples.shape[1] == cols


def test_kde_sampler_from_prior():
    ks = KDESampler()
    ks.from_prior(prior)
    samples = ks.draw(n)
    assert type(samples) is np.ndarray
    assert samples.shape[0] == n
    assert samples.shape[1] == cols


def test_uniform_sampler():
    low = np.array([0.0 for i in range(cols)])
    high = np.array([1.0 for i in range(cols)])
    us = UniformSampler(low=low, high=high)
    samples = us.draw(n)
    assert type(samples) is np.ndarray
    assert samples.shape[0] == n
    assert samples.shape[1] == cols
    assert samples.min() >= 0
    assert samples.max() <= 1


def test_uniform_sampler_from_prior():
    us = UniformSampler()
    us.from_prior(prior)
    samples = us.draw(n)
    assert type(samples) is np.ndarray
    assert samples.shape[0] == n
    assert samples.shape[1] == cols
    assert samples.min() >= 0
    assert samples.max() <= 1