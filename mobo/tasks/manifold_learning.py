from mobo.engines import Task
from mobo.manifold_learning import moboTSNE


class ManifoldTaskTSNE(Task):

    def __init__(self, kwargs, parallel=False, target=None):
        if target is None:
            target = self.learn_manifold
        super().__init__(parallel=parallel,
                         target=target,
                         kwargs=kwargs)

    def learn_manifold(self, normalized_data):
        """
        :param normalized_data: array-like object that has hopefully been normalized
        """
        print("TSNE: {}".format(normalized_data.shape))
        tsne = moboTSNE()
        tsne_cols = tsne.fit_transform(normalized_data)
        self.set_persistent(key='tsne_columns',
                            value=tsne_cols)
