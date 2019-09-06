import numpy as np


class BaseErrorCalculator(object):
    """Abstract base class for ErrorCalculators."""
    def calculate(self, actual, target):
        raise NotImplementedError()


class AbsoluteErrorCalculator(BaseErrorCalculator):
    """Calculator of absolute error."""
    def __init__(self) -> None:
        super().__init__()

    def calculate(self, actual: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Calculates the absolute difference between two arrays.
        
        Args:
            actual: Array of actual values.
            target: Array of target values.
        """
        return np.absolute(actual - target)


class LogCoshErrorCalculator(BaseErrorCalculator):
    """Calculator of log cosh error."""
    def __init__(self) -> None:
        super().__init__()

    def calculate(self, actual: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Calculates the log cosh error between two arrays.
        
        Args:
            actual: Array of actual values.
            target: Array of target values.
        """ 
        return np.log(np.cosh(actual - target))


class SquaredErrorCalculator(BaseErrorCalculator):
    """Calculator of squared error."""
    def __init__(self) -> None:
        super().__init__()

    def calculate(self, actual: np.ndarray, target: np.ndarray) -> np.ndarray:
        """ Calculates the squared error between two arrays.

        Args:
            actual: Array of actual values.
            target: Array of target values.
        """
        return (actual - target)**2
