"""
Customized implementations of clustering techniques
"""

from sklearn.cluster import DBSCAN


class moboDBSCAN(DBSCAN):

    def __init__(self, config):
        d = config['clustering']['dbscan']['args']
        kwargs = {}

        # process dbscan arguments
        for k, v in d:
            kwargs[k] = v

        # make sure eps was set by user or this custom method
        if 'eps' not in kwargs:
            kwargs['eps'] = self._calculate_eps()

        # initialize the base with the preset args
        super().__init__(**kwargs)

    # TODO
    def _calculate_eps(self):
        return 0.5
