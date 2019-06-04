

class Parameter(object):
    """Implementation of a continuous variable with optional constraints.

    Args:
        name (str): Name of the parameter.
        low (optional) (float): Minimum acceptable value.
        high (optional) (float): Maximum acceptable value.
        fixed (optional) (float): Fixed value for constant parameters.
    """

    def __init__(self, name, low=None, high=None, fixed=None):
        assert type(name) is str
        assert all((type(low) in [float, NoneType],
                    type(high) in [float, NoneType],
                    type(fixed) in [float, NoneType]))
        if type(low) is float and type(high) is float:
            assert low < high
        self._name = name,
        self._low = low
        self._high = high
        self._fixed = fixed

    @property
    def name(self):
        return self._name

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def fixed(self):
        return self._fixed


