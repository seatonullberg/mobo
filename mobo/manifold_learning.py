"""
Customized implementations of manifold learning techniques
"""

from sklearn.manifold import TSNE
import yaml


class moboTSNE(TSNE):

    def __init__(self):
        d = self.configuration['manifold_learning']['tsne']['args']
        kwargs = {}

        # process tsne arguments
        for k, v in d:
            kwargs[k] = v

        # TODO: custom param setting here

        super().__init__(**kwargs)

    @property
    def configuration(self):
        try:
            configuration = yaml.load(open("configuration.yaml"))
        except FileNotFoundError:
            # should be custom error
            raise

        return configuration
