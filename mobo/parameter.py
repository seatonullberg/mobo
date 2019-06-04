

class Parameter(object):
    """Implementation of a continuous variable with optional constraints.

    Args:
        name (str): Name of the parameter.
        lower (optional) (float): Minimum acceptable value.
        upper (optional) (float): Maximum acceptable value.
        fixed (optional) (float): Fixed value for constant parameters.
    """

    def __init__(self, name, lower=None, upper=None, fixed=None):
        assert type(name) is str
        assert all((type(lower) in [float, NoneType],
                    type(upper) in [float, NoneType],
                    type(fixed) in [float, NoneType]))
        self._name = name,
        self._lower = lower
        self._upper = upper
        self._fixed = fixed

    @property
    def name(self):
        return self._name

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def fixed(self):
        return self._fixed


