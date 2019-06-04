from mobo.filter import ParetoFilter, PercentileFilter
from mobo.filter import IntersectionalFilterSet, SequentialFilterSet
import numpy as np


costs = np.random.normal(size=(100, 3))


def test_pareto_filter():
    pf = ParetoFilter()
    mask = pf.apply(costs)
    filtered_data = costs[mask]
    assert filtered_data.shape[0] < costs.shape[0]


def test_percentile_filter():
    pf = PercentileFilter(percentile=5)
    mask = pf.apply(costs)
    filtered_data = costs[mask]
    assert 4 <= filtered_data.shape[0] <= 6


def test_intersectional_filter_set():
    # 0 filters
    filters = []
    ifs = IntersectionalFilterSet(filters=filters)
    result0 = ifs.apply(costs)
    assert result0.shape == costs.shape  # no change
    # 1 filter
    filters = [PercentileFilter(95)]
    ifs = IntersectionalFilterSet(filters=filters)
    result1 = ifs.apply(costs)
    assert 94 <= result1.shape[0] <= 96
    # 2 filters
    filters = [PercentileFilter(95), ParetoFilter()]
    ifs = IntersectionalFilterSet(filters=filters)
    result2 = ifs.apply(costs)
    assert result2.shape[0] < result1.shape[0]

def test_sequantial_filter_set():
    # 0 filters
    filters = []
    sfs = SequentialFilterSet(filters=filters)
    result0 = sfs.apply(costs)
    assert result0.shape == costs.shape  # no change
    # 1 filter
    filters = [PercentileFilter(95)]
    sfs = SequentialFilterSet(filters=filters)
    result1 = sfs.apply(costs)
    assert 94 <= result1.shape[0] <= 96
    # 2 filters
    filters = [PercentileFilter(95), ParetoFilter()]
    sfs = SequentialFilterSet(filters=filters)
    result2 = sfs.apply(costs)
    assert result2.shape[0] < result1.shape[0]
