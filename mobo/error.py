import numpy as np


class BaseErrorCalculator(object):
    """Representation of a general error calculator."""

    def __init__(self):
        pass

    def calculate(self, *args, **kwargs):
        err = ("{} does not implement the required `calculate` method."
               .format(self.__class__.__name__))
        raise NotImplementedError(err)


class AbsoluteErrorCalculator(BaseErrorCalculator):
    """Implementation of the absolute error calculator."""

    def __init__(self):
        super().__init__()

    def calculate(self, a, b):
        """Calculates the absolute error between two arrays.

        Args:
            a (numpy.ndarray): The first array.
            b (numpy.ndarray): The second array.

        Returns:
            numpy.ndarray
        """
        assert all((type(a) is np.ndarray,
                    type(b) is np.ndarray))
        assert a.shape == b.shape
        return np.absolute(a - b)


class LogCoshErrorCalculator(BaseErrorCalculator):
    """Implementation of the log cosh error calculator."""

    def __init__(self):
        super().__init__()

    def calculate(self, a, b):
        """Calculates the log cosh error between two arrays.

        Args:
            a (numpy.ndarray): The first array.
            b (numpy.ndarray): The second array.

        Returns:
            numpy.ndarray
        """
        assert all((type(a) is np.ndarray,
                    type(b) is np.ndarray))
        assert a.shape == b.shape
        return np.log(np.cosh(a - b))


class SquaredErrorCalculator(BaseErrorCalculator):
    """Implementation of the squared error calculator."""

    def __init__(self):
        super().__init__()

    def calculate(self, a, b):
        """Calculates the squared error between two arrays.

        Args:
            a (numpy.ndarray): The first array.
            b (numpy.ndarray): The second array.

        Returns:
            numpy.ndarray
        """
        assert all((type(a) is np.ndarray,
                    type(b) is np.ndarray))
        assert a.shape == b.shape
        return (a - b)**2
