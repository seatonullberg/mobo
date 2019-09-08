from mobo.data import OptimizationData
import numpy as np
from typing import Callable, List


class BaseFilter(object):
    """Abstract base class for Filters."""
    def apply(self, error):
        raise NotImplementedError()


class ParetoFilter(BaseFilter):
    """Pareto optimality filter."""
    def __init__(self):
        super().__init__()

    def apply(self, error: np.ndarray) -> np.ndarray:
        """Returns a mask of pareto efficient points.

        Args:
            error: Array of error values.
        """
        mask = np.ones(error.shape[0], dtype=bool)
        for i, err in enumerate(error):
            if mask[i]:
                # keep points with a lower error
                mask[mask] = np.any(error[mask] < err, axis=1)
                # and keep self
                mask[i] = True
        return mask


class PercentileFilter(BaseFilter):
    """Percentile rank filter.

    Args:
        cost_function: Function to calculate costs.
        percentile: The percentile rank to accept.
    """
    def __init__(self, cost_function: Callable, percentile: int) -> None:
        self.cost_function = cost_function
        self.percentile = percentile

    def apply(self, error: np.ndarray) -> np.ndarray:
        """Returns a mask of points ranking at or above the given percentile.

        Args:
            error: Array of error values.
        """
        costs = self.cost_function(error)
        critical_value = np.percentile(costs, self.percentile)
        return np.array([c <= critical_value for c in costs])


class BaseFilterSet(object):
    """Abstract base class for FilterSets."""
    def apply(self, data):
        raise NotImplementedError()


# TODO: Implement with OptimizationData


class IntersectionalFilterSet(BaseFilterSet):
    """Filter set which applies all filters simultaneously.

    Args:
        filters: Filters to apply.
    """
    def __init__(self, filters: List[BaseFilter]) -> None:
        self.filters = filters

    def apply(self, data: OptimizationData) -> None:
        """Applies all filters simultaneously.

        Args:
            data: OptimizationData.
        """
        masks = [f.apply(data.error_values) for f in self.filters]
        result_mask = masks[0]
        for mask in masks[1:]:
            result_mask = np.logical_and(result_mask, mask)
        data.drop(np.where(result_mask)[0])


class SequentialFilterSet(BaseFilterSet):
    """Filter set which applies all filters in sequence.

    Args:
        filters: Filters to apply.
    """
    def __init__(self, filters: List[BaseFilter]) -> None:
        self.filters = filters

    def apply(self, data: OptimizationData) -> None:
        """Applies all filters in sequence.

        Args:
            data: OptimizationData.
        """
        for f in self.filters:
            mask = f.apply(data.error_values)
            data.drop(np.where(mask)[0])
