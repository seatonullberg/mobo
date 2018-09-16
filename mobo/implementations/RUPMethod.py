from mobo.engines import IterativeTaskEngine, Task
from mobo.tasks.clustering import ClusterTaskDBSCAN
from mobo.tasks.file_operations import MergeFilesTask
from mobo.tasks.manifold_learning import ManifoldTaskTSNE
from mobo.tasks.monte_carlo_sampling import MonteCarloTask
from mobo.tasks.normalization import StandardNormalizationTask
from mobo.tasks.pareto import FilterParetoTask

import pandas as pd
import os


class RUPMethod(IterativeTaskEngine):

    def __init__(self, df, data_dir, evaluator, n_iterations):
        '''
        :param df: pandas.DataFrame object containing the input data
        :param data_dir: the path to the directory in which data files will be written
        '''
        assert type(df) == pd.DataFrame
        assert type(data_dir) == str
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        self.df = df
        self.data_dir = data_dir
        super().__init__(evaluator=evaluator, n_iterations=n_iterations)

    # TODO
    def init_from_file(self, filepath, data_dir):
        '''
        :param filepath: path to an input data file
        :param data_dir: the directory to which data files should be written
        '''
        pass

    def start(self, kwargs):
        task_list = [StandardNormalizationTask,
                     ManifoldTaskTSNE,
                     ClusterTaskDBSCAN,
                     MonteCarloTask,
                     FilterParetoTask,
                     MergeFilesTask]
        for i, task in enumerate(task_list):
            self.add_task(task(index=i))

        super().start(kwargs=kwargs)
