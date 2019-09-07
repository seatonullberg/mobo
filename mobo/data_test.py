from mobo.data import OptimizationData
from mobo.parameter import Parameter
from mobo.qoi import QoI
import numpy as np


def test_optimization_data_append_and_drop():
    parameters = [Parameter(name="test_parameter")]
    qois = [QoI(name="test_qoi", target=1.0, evaluator=lambda x: x)]
    data = OptimizationData(parameters, qois)
    iteration = 1
    p_values = np.array([1.0, 1.0])
    q_values = np.array([1.0, 1.0])
    e_values = np.array([0.0, 0.0])
    # append test
    data.append(iteration,
                p_values,
                q_values,
                e_values)
    assert np.array_equal(data.parameter_values, p_values)
    assert np.array_equal(data.qoi_values, q_values)
    assert np.array_equal(data.error_values, e_values)
    ids = np.array(["1_0", "1_1"])
    assert np.array_equal(data.ids, ids)
    for cid in data.cluster_ids:
        assert np.isnan(cid)
    # drop test
    data.drop(np.array([0]))
    new_size = 1
    assert data.parameter_values.size == new_size
    assert data.qoi_values.size == new_size
    assert data.error_values.size == new_size
    assert data.ids.size == new_size
    assert data.cluster_ids.size == new_size
