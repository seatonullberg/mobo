from mobo.configuration import Configuration
from mobo.engines import TaskEngine, moboKey
from mobo.tasks.normalization import StandardNormalizationTask
from mobo.tasks.manifold_learning import ManifoldTaskTSNE
from mobo.tasks.clustering import ClusterTaskDBSCAN, ClusterTaskKmeans
from mobo.tasks.matrix_operations import GroupByColumnValue, ConcatenateDataTask
from mobo.tasks.kde import KDEBandwidthTask
from mobo.tasks.monte_carlo_sampling import KDEMonteCarloTask

import pandas as pd
import numpy as np


def evaluate():
    pass


if __name__ == "__main__":
    # make a configuration
    config = Configuration()
    config = config.default

    rand_arr = np.random.randn(50, 5)

    df = pd.DataFrame(data=rand_arr)

    # load engine
    engine = TaskEngine(evaluator=evaluate)

    # init tasks
    d = {'data': df}
    normalize = StandardNormalizationTask(index=0,
                                          kwargs=d)

    d = {'normalized_data': moboKey('normalized_data')}
    tsne = ManifoldTaskTSNE(index=1,
                            kwargs=d)

    d = {'data': moboKey('tsne_columns')}
    cluster = ClusterTaskDBSCAN(index=2,
                                kwargs=d)

    d = {'a1': moboKey('normalized_data'),
         'a2': moboKey('dbscan_labels'),
         'axis': 1}
    concat = ConcatenateDataTask(index=3,
                                 kwargs=d)

    d = {'data': moboKey('concatenated_data'),
         'col_id': -1,
         'drop_selector': True}
    group = GroupByColumnValue(index=4,
                               kwargs=d)

    d = {'data': moboKey('grouped_subselections')}
    bandwidth = KDEBandwidthTask(index=5,
                                 kwargs=d)

    d = {'data': moboKey('grouped_subselections'),
         'bandwidth': moboKey('kde_bandwidth'),
         'n_samples': 100}
    sample = KDEMonteCarloTask(index=6,
                               kwargs=d)

    # add tasks
    engine.add_task(normalize)
    engine.add_task(tsne)
    engine.add_task(cluster)
    engine.add_task(concat)
    engine.add_task(group)
    engine.add_task(bandwidth)
    engine.add_task(sample)
    engine.start()
