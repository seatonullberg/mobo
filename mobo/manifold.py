from sklearn.manifold import MDS, TSNE
import numpy as np
from typing import Callable, Optional, Union


class BaseManifoldEmbedder(object):
    """Abstract base class."""
    def embed(self, data: np.ndarray):
        raise NotImplementedError()


class MdsManifoldEmbedder(BaseManifoldEmbedder):
    """Multidimensional scaling.
    
    Note:
        `n_jobs` parameter is intentionally ignored to prevent MPI problems.

    Args:
        Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html#sklearn.manifold.MDS
    """
    def __init__(self,
                 n_components: int = 2,
                 metric: bool = True,
                 n_init: int = 4,
                 max_iter: int = 300,
                 verbose: int = 0,
                 eps: float = 1e-3,
                 random_state: Optional[int] = None,
                 dissimilarity: str = "euclidean") -> None:
        n_jobs = None
        self.mds = MDS(n_components, metric, n_init, max_iter, verbose, eps,
                       n_jobs, random_state, dissimilarity)

    def embed(self, data: np.ndarray) -> np.ndarray:
        return self.mds.fit_transform(data)


class TsneManifoldEmbedder(BaseManifoldEmbedder):
    """t-distributed Stochastic Neighbor Embedding.
    
    Args:
        Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    """
    def __init__(self,
                 n_components: int = 2,
                 perplexity: float = 30.0,
                 early_exaggeration: float = 12.0,
                 learning_rate: float = 200.0,
                 n_iter: int = 1000,
                 n_iter_without_progress: int = 300,
                 min_grad_norm: float = 1e-7,
                 metric: Union[str, Callable] = "euclidean",
                 init: Union[str, np.ndarray] = "random",
                 verbose: int = 0,
                 random_state: Optional[int] = None,
                 method: str = "barnes_hut",
                 angle: float = 0.5) -> None:
        self.tsne = TSNE(n_components, perplexity, early_exaggeration,
                         learning_rate, n_iter, n_iter_without_progress,
                         min_grad_norm, metric, init, verbose, random_state,
                         method, angle)

    def embed(self, data: np.ndarray) -> np.ndarray:
        return self.tsne.fit_transform(data)
