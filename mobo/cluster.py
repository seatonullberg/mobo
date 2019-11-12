from abc import ABC
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from typing import Any, Callable, Optional, Union


class BaseClusterer(ABC):
    """Abstract base class for Clusterers."""
    def __init__(self, clusterer: Any) -> None:
        self._clusterer = clusterer

    def __call__(self, data: np.ndarray) -> np.ndarray:
        pass


class DbscanClusterer(BaseClusterer):
    """DBSCAN clustering technique.
    
    Notes:
        `n_jobs` parameter is intentionally ignored to prevent MPI problems.

    Args:
        Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    """
    def __init__(self,
                 eps: float = 0.5,
                 min_samples: int = 5,
                 metric: Union[str, Callable] = "euclidean",
                 metric_params: Optional[dict] = None,
                 algorithm: str = "auto",
                 leaf_size: int = 30,
                 p: Optional[float] = None) -> None:
        clusterer = DBSCAN(eps=eps,
                           min_samples=min_samples,
                           metric=metric,
                           metric_params=metric_params,
                           algorithm=algorithm,
                           leaf_size=leaf_size,
                           p=p,
                           n_jobs=None)
        super().__init__(clusterer)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self._clusterer.fit_predict(data)


class KmeansClusterer(BaseClusterer):
    """KMeans clustering technique.

    Notes:
        `n_jobs` parameter is intentionally ignored to prevent MPI problems.
    
    Args:
        Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """
    def __init__(self,
                 n_clusters: int = 8,
                 init: Union[str, np.ndarray] = "k-means++",
                 n_init: int = 10,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 precompute_distances: Union[str, bool] = "auto",
                 verbose: int = 0,
                 random_state: Union[int, None] = None,
                 copy_x: bool = True,
                 algorithm: str = "auto") -> None:
        clusterer = KMeans(n_clusters=n_clusters,
                           init=init,
                           n_init=n_init,
                           max_iter=max_iter,
                           tol=tol,
                           precompute_distances=precompute_distances,
                           verbose=verbose,
                           random_state=random_state,
                           copy_x=copy_x,
                           algorithm=algorithm,
                           n_jobs=None)
        super().__init__(clusterer)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self._clusterer.fit_predict(data)
