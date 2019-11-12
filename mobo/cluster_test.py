from mobo.cluster import DbscanClusterer, KmeansClusterer
import numpy as np

NROWS = 1000
DATA = np.random.normal(size=(NROWS, 3))


def test_dbscan_clusterer():
    dbscan = DbscanClusterer()
    cluster_ids = dbscan(DATA)
    assert cluster_ids.shape == (NROWS, )


def test_kmeans_clusterer():
    kmeans = KmeansClusterer()
    cluster_ids = kmeans(DATA)
    assert cluster_ids.shape == (NROWS, )
