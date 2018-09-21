from mobo.configuration import Configuration
from mobo.engines import TaskEngine, moboKey, Fork, Join
from mobo.tasks.normalization import StandardNormalizationTask
from mobo.tasks.manifold_learning import ManifoldTaskTSNE
from mobo.tasks.clustering import ClusterTaskDBSCAN, ClusterTaskKmeans
from mobo.tasks.matrix_operations import GroupByColumnValue, ConcatenatePairTask, ConcatenateListTask
from mobo.tasks.kde import KDEBandwidthTask
from mobo.tasks.monte_carlo_sampling import KDEMonteCarloTask
from mobo.tasks.evaluation import RootMeanSquaredErrorTask

import pandas as pd
import numpy as np


if __name__ == "__main__":
    # make a configuration
    config = Configuration()
    config = config.default

    rand_arr = np.random.randn(100, 5)

    df = pd.DataFrame(data=rand_arr)

    # load engine
    engine = TaskEngine()

    # normalize the data
    d = {'data': df}
    normalize = StandardNormalizationTask(kwargs=d)
    engine.add_component(normalize)

    # learn the tSNE manifold
    d = {'normalized_data': moboKey('normalized_data')}
    tsne = ManifoldTaskTSNE(kwargs=d)
    engine.add_component(tsne)

    # cluster the data by tSNE dimensions
    d = {'data': moboKey('tsne_columns')}
    cluster = ClusterTaskKmeans(kwargs=d)
    engine.add_component(cluster)

    # concat normal data with cluster labels
    d = {'a1': moboKey('normalized_data'),
         'a2': moboKey('kmeans_labels'),
         'axis': 1}
    concat = ConcatenatePairTask(kwargs=d)
    engine.add_component(concat)

    # subselect the concatenated data into groups by cluster id
    d = {'data': moboKey('concatenated_data'),
         'col_id': -1,
         'drop_selector': True}
    group = GroupByColumnValue(kwargs=d)
    engine.add_component(group)

    # fork to calculate the bandwidth of each group
    d = {'data': moboKey('grouped_subselections')}
    bandwidth = KDEBandwidthTask(kwargs={})
    bandwidth_fork = Fork(iterators=d, task=bandwidth)
    engine.add_component(bandwidth_fork)

    # remain forked to do kde sampling
    d = {'data': moboKey('grouped_subselections')}
    sample = KDEMonteCarloTask(kwargs={'bandwidth': moboKey('kde_bandwidth'),
                                       'n_samples': 50})
    sample_fork = Fork(iterators=d, task=sample)
    engine.add_component(sample_fork)

    '''
    # to calculate errors
    d = {'actual': rand_arr,
         'experimental': moboKey('kde_samples')}
    error = RootMeanSquaredErrorTask(kwargs=d)
    '''
    engine.start()
    # TODO: fix persistent data overriding issue
    # occurs in KDEMonteCarloTask because the bandwidth
    # gets overwritten which means it is no longer working on an iterable
