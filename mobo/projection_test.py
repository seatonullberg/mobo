from mobo.projection import MDSProjector, PCAProjector, TSNEProjector
import numpy as np

NROWS = 100
DATA = np.random.normal(size=(NROWS, 3))


def test_mds_projector():
    mds = MDSProjector()
    projection = mds(DATA)
    assert projection.shape[0] == NROWS


def test_pca_projector():
    pca = PCAProjector()
    projection = pca(DATA)
    assert projection.shape[0] == NROWS


def test_tsne_projector():
    tsne = TSNEProjector()
    projection = tsne(DATA)
    assert projection.shape[0] == NROWS
