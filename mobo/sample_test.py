from mobo.sample import GaussianSampler, KDESampler, UniformSampler
import numpy as np
from scipy.stats import gaussian_kde

n = 10
rows = 100
cols = 3


def test_gaussian_sampler():
    mean = np.array([0.0 for i in range(cols)])
    std = np.array([1.0 for i in range(cols)])
    gs = GaussianSampler(mean=mean, std=std)
    samples = gs.draw(n)
    assert type(samples) is np.ndarray
    assert samples.shape[0] == n
    assert samples.shape[1] == cols


def test_gaussian_sampler_from_prior():
    prior = np.random.uniform(size=(rows, cols))
    gs = GaussianSampler.from_prior(prior)
    samples = gs.draw(n)
    assert type(samples) is np.ndarray
    assert samples.shape[0] == n
    assert samples.shape[1] == cols


def test_kde_sampler():
    arr = np.random.normal(size=(100, cols))
    bw = 0.1
    ks = KDESampler(bw, arr)
    samples = ks.draw(n)
    assert type(samples) is np.ndarray
    assert samples.shape[0] == n
    assert samples.shape[1] == cols


def test_kde_sampler_from_prior():
    prior = np.random.uniform(size=(rows, cols))
    prior = gaussian_kde(prior)
    ks = KDESampler.from_prior(prior)
    samples = ks.draw(n)
    assert type(samples) is np.ndarray
    assert samples.shape[0] == n
    assert samples.shape[1] == cols


def test_uniform_sampler():
    low = np.array([0.0 for i in range(cols)])
    high = np.array([1.0 for i in range(cols)])
    us = UniformSampler(low, high)
    samples = us.draw(n)
    assert type(samples) is np.ndarray
    assert samples.shape[0] == n
    assert samples.shape[1] == cols
    assert samples.min() >= 0
    assert samples.max() <= 1


def test_uniform_sampler_from_prior():
    prior = np.random.uniform(size=(rows, cols), low=0, high=1)
    us = UniformSampler.from_prior(prior)
    samples = us.draw(n)
    assert type(samples) is np.ndarray
    assert samples.shape[0] == n
    assert samples.shape[1] == cols
    assert samples.min() >= 0
    assert samples.max() <= 1
