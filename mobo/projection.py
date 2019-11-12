from abc import ABC
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from typing import Callable, Optional, Union


class BaseProjector(ABC):
    """Abstract base class for Projectors."""
    def __call__(self, data: np.ndarray) -> np.ndarray: 
        pass


class MDSProjector(BaseProjector):
    """Multi-Dimensional Scaling projector.
    
    Args:
        Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html#sklearn.manifold.MDS
    """
    def __init__(self,
                 metric: bool = True,
                 n_init: int = 4,
                 max_iter: int = 300,
                 verbose: int = 0,
                 eps: float = 1e-3,
                 random_state: Optional[int] = None,
                 dissimilarity: str = "euclidean") -> None:
        self._projector = MDS(n_components=2,
                              n_jobs=None,
                              metric=metric,
                              n_init=n_init,
                              max_iter=max_iter,
                              verbose=verbose,
                              eps=eps,
                              random_state=random_state,
                              dissimilarity=dissimilarity)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self._projector.fit_transform(data)


class PCAProjector(BaseProjector):
    """Principal Component Analysis projector.
    
    Args:
        Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """
    def __init__(self,
                 whiten: bool = False,
                 svd_solver: str = "auto",
                 tol: float = 0.0,
                 iterated_power: Union[str, int] = "auto",
                 random_state: Optional[int] = None) -> None:
        self._projector = PCA(n_components=2,
                              whiten=whiten,
                              svd_solver=svd_solver,
                              tol=tol,
                              iterated_power=iterated_power,
                              random_state=random_state)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self._projector.fit_transform(data)


class TSNEProjector(BaseProjector):
    """t-Distributed Stochastic Neighbor Embedding projector.
    
    Args:
        Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    """
    def __init__(self,
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
        self._projector = TSNE(n_components=2,
                               perplexity=perplexity,
                               early_exaggeration=early_exaggeration,
                               learning_rate=learning_rate,
                               n_iter=n_iter,
                               n_iter_without_progress=n_iter_without_progress,
                               min_grad_norm=min_grad_norm,
                               metric=metric,
                               init=init,
                               verbose=verbose,
                               random_state=random_state,
                               method=method,
                               angle=angle)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self._projector.fit_transform(data)
