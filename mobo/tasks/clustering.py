from mobo.engines import Task
from mobo.clustering import moboDBSCAN, moboKmeans

import numpy as np


class ClusterTaskDBSCAN(Task):

    def __init__(self, kwargs, parallel=False, target=None):
        if target is None:
            target = self.cluster
        super().__init__(parallel=parallel,
                         target=target,
                         kwargs=kwargs)

    def cluster(self, data):
        """
        :param data: array-like object
        """
        dbscan = moboDBSCAN()
        labels = dbscan.fit_predict(data)
        labels = np.resize(labels, (labels.shape[0], 1))
        self.set_persistent(key='dbscan_labels',
                            value=labels)


class ClusterTaskKmeans(Task):

    def __init__(self, kwargs, parallel=False, target=None):
        if target is None:
            target = self.cluster
        super().__init__(parallel=parallel,
                         target=target,
                         kwargs=kwargs)

    def cluster(self, data):
        """
        :param data: array-like object
        """
        print("KMEANS: {}".format(data.shape))
        kmeans = moboKmeans()
        labels = kmeans.fit_predict(data)
        labels = np.resize(labels, (labels.shape[0], 1))
        self.set_persistent(key='kmeans_labels',
                            value=labels)
