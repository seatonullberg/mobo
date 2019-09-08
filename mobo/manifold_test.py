from mobo.manifold import MdsManifoldEmbedder, TsneManifoldEmbedder
import numpy as np


def test_mds_manifold_embedder():
    nrows = 100
    data = np.random.normal(size=(nrows, 3))
    embedder = MdsManifoldEmbedder()
    embedding = embedder.embed(data)
    assert embedding.shape[0] == nrows


def test_tsne_manifold_embedder():
    nrows = 100
    data = np.random.normal(size=(nrows, 3))
    embedder = TsneManifoldEmbedder()
    embedding = embedder.embed(data)
    assert embedding.shape[0] == nrows
