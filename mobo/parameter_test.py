from mobo.parameter import Parameter
import pytest


def test_parameter():
    with pytest.raises(ValueError):
        _ = Parameter(name="test", lower_bound=1.0, upper_bound=0.0)
