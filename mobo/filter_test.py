from mobo.data import OptimizationData
from mobo.filter import ParetoFilter, PercentileFilter
from mobo.filter import IntersectionalFilterSet, SequentialFilterSet
from mobo.parameter import Parameter
from mobo.qoi import QoI
import numpy as np


def test_pareto_filter():
    pf = ParetoFilter()
    costs = np.random.normal(size=(100, 3))
    mask = pf.apply(costs)
    filtered_data = costs[mask]
    assert filtered_data.shape[0] < costs.shape[0]


def test_percentile_filter():
    pf = PercentileFilter(cost_function=lambda x: x.sum(axis=1), percentile=5)
    error = np.random.normal(size=(100, 3))
    mask = pf.apply(error)
    error = error[mask]
    assert error.shape[0] == 5


def test_intersectional_filter_set():
    pareto_filter = ParetoFilter()
    percentile_filter = PercentileFilter(cost_function=lambda x: x.sum(axis=1),
                                         percentile=25)
    filters = [pareto_filter, percentile_filter]
    filter_set = IntersectionalFilterSet(filters)
    # mock optimization data
    nrows = 1000
    p_values = np.random.normal(size=(nrows, 3))
    q_values = np.random.normal(size=(nrows, 2))
    e_values = np.random.normal(size=(nrows, 2))
    parameters = [Parameter("p0"), Parameter("p1"), Parameter("p2")]
    evaluator = lambda x: x
    qois = [QoI(evaluator, "q0", 1.0), QoI(evaluator, "q1", 1.0)]
    data = OptimizationData(parameters, qois)
    data.append(1, p_values, q_values, e_values)
    filter_set.apply(data)
    # ensure filtering has been done
    assert 0 < data.parameter_values.shape[0] < nrows


def test_sequential_filter_set():
    pareto_filter = ParetoFilter()
    percentile_filter = PercentileFilter(cost_function=lambda x: x.sum(axis=1),
                                         percentile=25)
    filters = [pareto_filter, percentile_filter]
    filter_set = SequentialFilterSet(filters)
    # mock optimization data
    nrows = 1000
    p_values = np.random.normal(size=(nrows, 3))
    q_values = np.random.normal(size=(nrows, 2))
    e_values = np.random.normal(size=(nrows, 2))
    parameters = [Parameter("p0"), Parameter("p1"), Parameter("p2")]
    evaluator = lambda x: x
    qois = [QoI(evaluator, "q0", 1.0), QoI(evaluator, "q1", 1.0)]
    data = OptimizationData(parameters, qois)
    data.append(1, p_values, q_values, e_values)
    filter_set.apply(data)
    # ensure filtering has been done
    assert 0 < data.parameter_values.shape[0] < nrows
