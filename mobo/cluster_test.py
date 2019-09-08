from mobo.cluster import DbscanClusterer, KmeansClusterer
from mobo.data import OptimizationData
from mobo.parameter import Parameter
from mobo.qoi import QoI
import numpy as np


def test_dbscan_clusterer():
    nrows = 1000
    data = np.random.normal(size=(nrows, 3))
    dbscan = DbscanClusterer()
    cluster_ids = dbscan.cluster(data)
    assert cluster_ids.shape == (nrows, )


def test_kmeans_clusterer():
    nrows = 1000
    data = np.random.normal(size=(nrows, 3))
    kmeans = KmeansClusterer()
    cluster_ids = kmeans.cluster(data)
    assert cluster_ids.shape == (nrows, )
