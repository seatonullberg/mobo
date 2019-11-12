from mobo.error import AbsoluteErrorCalculator, LogCoshErrorCalculator
from mobo.error import RawErrorCalculator, SquaredErrorCalculator
import numpy as np

A = np.random.normal(size=(1000, 3))
B = np.random.normal(size=(1000, 3))


def test_absolute_error_calculator():
    absolute = AbsoluteErrorCalculator()
    err = absolute(A, B)
    assert np.max(err) >= 0
    assert err.shape == A.shape == B.shape


def test_log_cosh_error_calculator():
    log_cosh = LogCoshErrorCalculator()
    err = log_cosh(A, B)
    assert err.shape == A.shape == B.shape


def test_raw_error_calculator():
    raw = RawErrorCalculator()
    err = raw(A, B)
    assert err.shape == A.shape == B.shape


def test_squared_error_calculator():
    squared = SquaredErrorCalculator()
    err = squared(A, B)
    assert np.max(err) >= 0
    assert err.shape == A.shape == B.shape
