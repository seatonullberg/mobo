import numpy as np


class BaseFilter(object):
    """Representation of a general data filter."""

    def __init__(self):
        pass

    def apply(self, *args, **kwargs):
        err = ("{} does not implement the required `apply` method."
               .format(self.__class__.__name__))
        raise NotImplementedError(err)


class ParetoFilter(BaseFilter):
    """Implementation of a Pareto optimal filter."""

    def __init__(self):
        super().__init__()

    def apply(self, arr):
        """Determines which points are Pareto optimal.

        Args:
            arr (numpy.ndarray): Array of costs.

        Returns:
            numpy.ndarray
            - boolean mask
        """
        assert type(arr) is np.ndarray
        is_efficient = np.ones(arr.shape[0], dtype=bool)
        for i, cost in enumerate(arr):
            if is_efficient[i]:
                # keep points with a lower cost
                is_efficient[is_efficient] = np.any(arr[is_efficient] < cost,
                                                    axis=1)
                # and keep self
                is_efficient[i] = True
        return is_efficient


class PercentileFilter(BaseFilter):
    """Implementation of a percentile cost filter.
       - Costs are calculated as row-wise summations.

    Args:
        percentile (int): The percentile value to accept.
    """

    def __init__(self, percentile):
        assert type(percentile) is int
        if not 0 < percentile < 100:
            err = "`percentile` must be between 0 and 100."
            raise ValueError(err)
        self._percentile = percentile

    @property
    def percentile(self):
        return self._percentile

    def apply(self, arr):
        """Determines which points score within a particular percentile.

        Args:
            arr (numpy.ndarray): Array of costs.

        Returns:
            numpy.ndarray
            - boolean mask
        """
        assert type(arr) is np.ndarray
        costs = np.sum(arr, axis=1)
        perc_val = np.percentile(costs, self.percentile)
        eff_ids = costs <= perc_val
        return eff_ids


class BaseFilterSet(object):
    """Representation of a general collection of filters.

    Args:
        filters (iterable of instance of BaseFilter): Filters to apply.
    """

    def __init__(self, filters):
        for f in filters:
            assert isinstance(f, BaseFilter)
        self._filters = filters

    @property
    def filters(self):
        return self._filters

    def apply(self):
        err = ("{} does not implement the required `apply` method."
               .format(self.__class__.__name__))
        raise NotImplementedError(err)


class IntersectionalFilterSet(BaseFilterSet):
    """Implementation of a filter set which applies all filters simultaneously.

    Args:
        filters (iterable of instance of BaseFilter): Filters to apply.
    """

    def __init__(self, filters):
        super().__init__(filters)

    def apply(self, arr):
        """Applies all filter masks together.

        Args:
            arr (numpy.ndarray): Array of costs.

        Returns:
            numpy.ndarray
        """
        assert type(arr) is np.ndarray
        masks = [f.apply(arr) for f in self.filters]
        if len(masks) == 0:
            # no filtering required
            res = arr
        elif len(masks) == 1:
            # apply the only mask
            res = arr[masks[0]]
        else:
            # apply intersection of all masks
            current_mask = masks[0]
            for m in masks[1:]:
                current_mask = np.logical_and(current_mask, m)
            res = arr[current_mask]
        return res


class SequentialFilterSet(BaseFilterSet):
    """Implementation of a filter set which applies all filters in sequence.

    Args:
        filters (iterable of instance of BaseFilter): Filters to apply.
    """

    def __init__(self, filters):
        super().__init__(filters)

    def apply(self, arr):
        """Applies all filter masks in sequence.

        Args:
            arr (numpy.ndarray): Array of costs.

        Returns:
            numpy.ndarray
        """
        assert type(arr) is np.ndarray
        for f in self.filters:
            mask = f.apply(arr)
            arr = arr[mask]
        return arr

