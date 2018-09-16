from mobo.engines import Task
from mobo.clustering import moboDBSCAN


class ClusterTaskDBSCAN(Task):

    def __init__(self, index):
        super().__init__(parallel=False,
                         index=index,
                         target=self.cluster)

    def cluster(self, df):
        '''
        :param df: pandas.DataFrame object
        :return: cluster id labels for each sample
        '''
        dbscan = moboDBSCAN(config=self.configuration)
        labels = dbscan.fit_predict(df)
        return labels
