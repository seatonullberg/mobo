"""
Customized implementations of manifold learning techniques
"""

from sklearn.manifold import TSNE


class moboTSNE(TSNE):

    def __init__(self):
        super().__init__()
