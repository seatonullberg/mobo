from mobo.engines import Task
from mobo.clustering import moboDBSCAN


class ClusterTaskDBSCAN(Task):

    def __init__(self, index):
        super().__init__(parallel=False,
                         index=index,
                         target=self.cluster)

    def cluster(self, data=None, data_key='tsne_columns'):
        '''
        :param data: array-like object
        :param data_key: str to retrieve data from persistent database
        '''
        if data is None:
            data = self.get_persistent(data_key)

        dbscan = moboDBSCAN(config=self.configuration)
        labels = dbscan.fit_predict(data)

        self.set_persistent(key='dbscan_column',
                            value=labels)
