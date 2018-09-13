"""
Customized implementations of clustering techniques
"""

from sklearn.cluster import DBSCAN


class moboDBSCAN(DBSCAN):

    def __init__(self):
        super().__init__()
