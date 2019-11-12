from abc import ABC
import numpy as np


class BaseErrorCalculator(ABC):
    """Abstract base class for ErrorCalculators."""
    def __call__(self, actual: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass


class AbsoluteErrorCalculator(BaseErrorCalculator):
    """Calculator of absolute error."""
    def __call__(self, actual: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.absolute(actual - target)


class LogCoshErrorCalculator(BaseErrorCalculator):
    """Calculator of log cosh error."""
    def __call__(self, actual: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.log(np.cosh(actual - target))


class RawErrorCalculator(BaseErrorCalculator):
    """Calculator of raw error."""
    def __call__(self, actual: np.ndarray, target: np.ndarray) -> np.ndarray:
        return actual - target


class SquaredErrorCalculator(BaseErrorCalculator):
    """Calculator of squared error."""
    def __call__(self, actual: np.ndarray, target: np.ndarray) -> np.ndarray:
        return (actual - target)**2
