from mobo.engines import Task
from mobo.clustering import moboDBSCAN


class ClusterTaskDBSCAN(Task):

    def __init__(self, index, data_key=None, args=None):
        super().__init__(parallel=False,
                         index=index,
                         target=self.cluster,
                         data_key=data_key,
                         args=args)

    def cluster(self, data):
        """
        :param data: array-like object
        """
        dbscan = moboDBSCAN(config=self.configuration)
        labels = dbscan.fit_predict(data)

        self.set_persistent(key='dbscan_labels',
                            value=labels)
