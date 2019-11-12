from mobo.filter import ParetoFilter, PercentileFilter, ZscoreFilter
import numpy as np

DATA = np.random.normal(size=(1000, 3))


def test_pareto_filter():
    pareto = ParetoFilter()
    mask = pareto(DATA)
    filtered_data = DATA[mask]
    assert filtered_data.shape[0] < DATA.shape[0]

def test_percentile_filter():
    perc = PercentileFilter() # 95th percentile default
    mask = perc(DATA)
    filtered_data = DATA[mask]
    assert filtered_data.shape[0] == 950

def test_zscore_filter():
    zscore = ZscoreFilter() # -1.5 z score default
    mask = zscore(DATA)
    filtered_data = DATA[mask]
    assert filtered_data.shape[0] < DATA.shape[0]
