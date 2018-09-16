from mobo.engines import Task
from sklearn.preprocessing import StandardScaler


class StandardNormalizationTask(Task):

    def __init__(self, index):
        super().__init__(parallel=False,
                         index=index,
                         target=self.normalize)

    def normalize(self, df):
        '''
        :param df: pandas.DataFrame object
        :return: normalized form (mean=0, variance=1) of the df
        '''
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        return df
