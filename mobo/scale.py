from mobo.data import OptimizationData
import numpy as np
from sklearn import preprocessing
from typing import Tuple


class BaseScaler(object):
    """Abstract base class for Scalers."""
    def scale(self, data: np.ndarray):
        raise NotImplementedError()


class RobustScaler(BaseScaler):
    """Robust scaling technique to compensate for outliers.
    
    Args:
        Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler
    """
    def __init__(self,
                 with_centering: bool = True,
                 with_scaling: bool = True,
                 quantile_range: Tuple[float, float] = (25.0, 75.0),
                 copy: bool = True) -> None:
        self.scaler = preprocessing.RobustScaler(with_centering, with_scaling,
                                                 quantile_range, copy)

    def scale(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(data)


class StandardScaler(BaseScaler):
    """Typical scaling technique to remove mean and scale by variance.
    
    Args:
        Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    """
    def __init__(self,
                 copy: bool = True,
                 with_mean: bool = True,
                 with_std: bool = True) -> None:
        self.scaler = preprocessing.StandardScaler(copy, with_mean, with_std)

    def scale(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(data)
