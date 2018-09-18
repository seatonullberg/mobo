"""
Customized implementations of clustering techniques
"""

from sklearn.cluster import DBSCAN, KMeans
import yaml


class moboDBSCAN(DBSCAN):

    def __init__(self):
        d = self.configuration['clustering']['dbscan']['args']
        kwargs = {}

        # process dbscan arguments
        for k, v in d.items():
            kwargs[k] = v

        # make sure eps was set by user or this custom method
        if 'eps' not in kwargs:
            kwargs['eps'] = self._calculate_eps()

        # initialize the base with the preset args
        super().__init__(**kwargs)

    # TODO
    def _calculate_eps(self):
        return 0.5

    @property
    def configuration(self):
        try:
            configuration = yaml.load(open("configuration.yaml"))
        except FileNotFoundError:
            # should be custom error
            raise

        return configuration


class moboKmeans(KMeans):

    def __init__(self):
        d = self.configuration['clustering']['kmeans']['args']
        kwargs = {}

        # process kmeans args
        for k, v in d.items():
            kwargs[k] = v

        super().__init__(**kwargs)

    @property
    def configuration(self):
        try:
            configuration = yaml.load(open("configuration.yaml"))
        except FileNotFoundError:
            # should be custom error
            raise

        return configuration
