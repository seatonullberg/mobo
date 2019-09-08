from mobo.error import AbsoluteErrorCalculator, LogCoshErrorCalculator, SquaredErrorCalculator
import numpy as np

a = np.random.normal(size=(999, 9))
b = np.random.normal(size=(999, 9))


def test_absolute_error():
    err = AbsoluteErrorCalculator().calculate(a, b)
    assert np.max(err) >= 0
    assert err.shape == a.shape == b.shape


def test_log_cosh_error():
    err = LogCoshErrorCalculator().calculate(a, b)
    assert err.shape == a.shape == b.shape


def test_squared_error():
    err = SquaredErrorCalculator().calculate(a, b)
    assert np.max(err) >= 0
    assert err.shape == a.shape == b.shape
