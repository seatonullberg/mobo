from mobo.scale import RobustScaler, StandardScaler
import numpy as np


def test_robust_scaler():
    data = np.random.normal(size=(100, 3))
    scaler = RobustScaler()
    scaled_data = scaler.scale(data)
    assert not np.array_equal(data, scaled_data)
    assert data.shape == scaled_data.shape


def test_standard_scaler():
    data = np.random.normal(size=(100, 3))
    scaler = StandardScaler()
    scaled_data = scaler.scale(data)
    assert not np.array_equal(data, scaled_data)
    assert data.shape == scaled_data.shape
