from mobo.error import AbsoluteError, LogCoshError, SquaredError


a, b = 1.0, 1.5


def test_absolute_error():
    ae = AbsoluteError()
    err = ae.calculate(a, b)
    assert err == 0.5


def test_log_cosh_error():
    lce = LogCoshError()
    err = lce.calculate(a, b)
    assert err == 0.12011450695827745


def test_squared_error():
    se = SquaredError()
    err = se.calculate(a, b)
    assert err == 0.25