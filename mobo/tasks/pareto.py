from mobo.engines import Task
from mobo.pareto import calculate_pareto

import pandas as pd


class FilterParetoTask(Task):

    def __init__(self, index):
        super().__init__(parallel=False,
                         index=index,
                         target=self.filter)

    def filter(self, df):
        '''
        :param df: pandas.DataFrame object
        :return: new DataFrame consisting of only non-dominated points
        '''
        # convert df to array from efficient calcs
        arr = df.as_matrix()
        mask = calculate_pareto(arr)
        arr = arr[mask]
        filtered_df = pd.DataFrame(data=arr,
                                   columns=list(df))
        return filtered_df
