from mobo.engines import Task
from mobo.clustering import moboDBSCAN

import numpy as np


class ClusterTaskDBSCAN(Task):

    def __init__(self, index, kwargs):
        super().__init__(parallel=False,
                         index=index,
                         target=self.cluster,
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
