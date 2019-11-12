from abc import ABC
import numpy as np


class BaseFilter(ABC):
    """Abstract base class for Filters."""
    def __call__(self, data: np.ndarray) -> np.ndarray:
        pass


class ParetoFilter(BaseFilter):
    """Pareto optimality filter."""
    def __call__(self, data: np.ndarray) -> np.ndarray:
        mask = np.ones(data.shape[0], dtype=bool)
        for i, err in enumerate(data):
            if mask[i]:
                # keep points with a lower error
                mask[mask] = np.any(data[mask] < err, axis=1)
                # and keep self
                mask[i] = True
        return mask
