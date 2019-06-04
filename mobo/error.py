import numpy as np


class BaseError(object):
    """Representation of a general error calculator."""

    def __init__(self):
        pass

    def calculate(self, *args, **kwargs):
        err = ("{} does not implement the required `calculate` method."
               .format(self.__class__.__name__))
        raise NotImplementedError(err)


class AbsoluteError(BaseError):
    """Implementation of the absolute error calculator."""

    def __init__(self):
        super().__init__()

    def calculate(self, a, b):
        """Calculates the absolute error between two scalars.

        Args:
            a (float): The first scalar.
            b (float): The second scalar.

        Returns:
            float
        """
        assert all((type(a) is float,
                    type(b) is float))
        return abs(a - b)


class LogCoshError(BaseError):
    """Implementation of the log cosh error calculator."""

    def __init__(self):
        super().__init__()

    def calculate(self, a, b):
        """Calculates the log cosh error between two scalars.

        Args:
            a (float): The first scalar.
            b (float): The second scalar.

        Returns:
            float
        """
        assert all((type(a) is float,
                    type(b) is float))
        return np.log(np.cosh(a - b))


class SquaredError(BaseError):
    """Implementation of the squared error calculator."""

    def __init__(self):
        super().__init__()

    def calculate(self, a, b):
        """Calculates the squared error between two scalars.

        Args:
            a (float): The first scalar.
            b (float): The second scalar.

        Returns:
            float
        """
        assert all((type(a) is float,
                    type(b) is float))
        return (a - b)**2

