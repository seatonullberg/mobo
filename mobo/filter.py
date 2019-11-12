from abc import ABC
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import normalize


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


class PercentileFilter(BaseFilter):
    """Percentile scoring filter."""
    def __init__(self, percentile: int = 95) -> None:
        self._percentile = percentile
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        normalized = np.absolute(normalize(data, axis=0))
        scores = np.sum(normalized, axis=1) # sum each row
        critical_score = np.percentile(scores, self._percentile)
        return np.array([s <= critical_score for s in scores])


class ZscoreFilter(BaseFilter):
    """Z-Score filter."""
    def __init__(self, z: float = -1.5) -> None:
        self._z = z

    def __call__(self, data: np.ndarray) -> np.ndarray:
        normalized = np.absolute(normalize(data, axis=0))
        scores = np.sum(normalized, axis=1) # sum each row
        z_values = zscore(scores)
        return np.array([z >= self._z for z in z_values])
