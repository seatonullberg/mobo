from mobo.filter import *
import numpy as np

DATA = np.random.normal(size=(1000, 3))


def test_pareto_filter():
    pareto = ParetoFilter()
    mask = pareto(DATA)
    filtered_data = DATA[mask]
    assert filtered_data.shape[0] < DATA.shape[0]
