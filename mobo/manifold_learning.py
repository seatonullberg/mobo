"""
Customized implementations of manifold learning techniques
"""

from sklearn.manifold import TSNE


class moboTSNE(TSNE):

    def __init__(self, config):
        d = config['manifold_learning']['tsne']['args']
        kwargs = {}

        # process tsne arguments
        for k, v in d:
            kwargs[k] = v

        # TODO: custom param setting here

        super().__init__()
