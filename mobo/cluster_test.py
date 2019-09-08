from mobo.cluster import DbscanClusterer, KmeansClusterer
from mobo.data import OptimizationData
from mobo.parameter import Parameter
from mobo.qoi import QoI
import numpy as np


def test_dbscan_clusterer():
    # mock optimization data
    nrows = 1000
    p_values = np.random.normal(size=(nrows, 3))
    q_values = np.random.normal(size=(nrows, 2))
    e_values = np.random.normal(size=(nrows, 2))
    parameters = [
        Parameter("p0"), Parameter("p1"), Parameter("p2")
    ]
    evaluator = lambda x: x
    qois = [
        QoI(evaluator, "q0", 1.0), QoI(evaluator, "q1", 1.0)
    ]
    data = OptimizationData(parameters, qois)
    data.append(1, p_values, q_values, e_values)
    # mock clusterer
    dbscan = DbscanClusterer()
    cluster_ids = dbscan.cluster(data)
    assert cluster_ids.shape == (nrows,)


def test_kmeans_clusterer():
    # mock optimization data
    nrows = 1000
    p_values = np.random.normal(size=(nrows, 3))
    q_values = np.random.normal(size=(nrows, 2))
    e_values = np.random.normal(size=(nrows, 2))
    parameters = [
        Parameter("p0"), Parameter("p1"), Parameter("p2")
    ]
    evaluator = lambda x: x
    qois = [
        QoI(evaluator, "q0", 1.0), QoI(evaluator, "q1", 1.0)
    ]
    data = OptimizationData(parameters, qois)
    data.append(1, p_values, q_values, e_values)
    # mock clusterer
    kmeans = KmeansClusterer()
    cluster_ids = kmeans.cluster(data)
    assert cluster_ids.shape == (nrows,)
