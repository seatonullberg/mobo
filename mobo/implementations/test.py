from mobo.configuration import Configuration
from mobo.engines import TaskEngine
from mobo.tasks.normalization import StandardNormalizationTask
from mobo.tasks.manifold_learning import ManifoldTaskTSNE
from mobo.tasks.clustering import ClusterTaskDBSCAN

import pandas as pd
import numpy as np


def evaluate():
    pass


if __name__ == "__main__":
    # make a configuration
    config = Configuration()
    config = config.default

    # fake data
    rand_arr = np.random.rand(100, 5)
    df = pd.DataFrame(data=rand_arr)

    # load engine
    engine = TaskEngine(evaluator=evaluate)

    # init tasks
    normalize = StandardNormalizationTask(index=0,
                                          args=df)
    tsne = ManifoldTaskTSNE(index=1,
                            data_key='normalized_data')
    cluster = ClusterTaskDBSCAN(index=2,
                                data_key='tsne_columns')

    # add tasks
    engine.add_task(normalize)
    engine.add_task(tsne)
    engine.add_task(cluster)
    engine.start()
