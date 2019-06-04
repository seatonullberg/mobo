from mobo.error import AbsoluteError, LogCoshError, SquaredError
import numpy as np


a = np.random.normal(size=(100, 3))
b = np.random.normal(size=(100, 3))


def test_absolute_error():
    ae = AbsoluteError()
    err = ae.calculate(a, b)
    assert type(err) is np.ndarray
    assert err.shape == a.shape == b.shape


def test_log_cosh_error():
    lce = LogCoshError()
    err = lce.calculate(a, b)
    assert type(err) is np.ndarray
    assert err.shape == a.shape == b.shape


def test_squared_error():
    se = SquaredError()
    err = se.calculate(a, b)
    assert type(err) is np.ndarray
    assert err.shape == a.shape == b.shape