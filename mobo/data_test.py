from mobo.data import OptimizationData
from mobo.parameter import Parameter
from mobo.qoi import QoI
import numpy as np


def test_optimization_data_append_and_drop():
    parameters = [Parameter(name="test_parameter")]
    qois = [QoI(name="test_qoi", target=1.0, evaluator=lambda x: x)]
    data = OptimizationData(parameters, qois)
    iteration = 1
    nrows = 100
    p_values = np.random.normal(size=(nrows, 1))
    q_values = np.random.normal(size=(nrows, 1))
    e_values = np.random.normal(size=(nrows, 1))
    # append test
    data.append(iteration,
                p_values,
                q_values,
                e_values)
    print(data.parameter_values.shape)
    print(p_values.shape)
    assert np.array_equal(data.parameter_values, p_values)
    assert np.array_equal(data.qoi_values, q_values)
    assert np.array_equal(data.error_values, e_values)
    assert data.ids[-1] == "1_{}".format(nrows-1)
    for cid in data.cluster_ids:
        assert np.isnan(cid)
    # drop test
    data.drop(np.array([0]))
    assert data.parameter_values.size == nrows - 1
    assert data.qoi_values.size == nrows - 1
    assert data.error_values.size == nrows - 1
    assert data.ids.size == nrows - 1
    assert data.cluster_ids.size == nrows - 1
