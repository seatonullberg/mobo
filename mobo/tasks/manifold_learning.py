from mobo.engines import Task
from mobo.manifold_learning import moboTSNE


class ManifoldTaskTSNE(Task):

    def __init__(self, index, kwargs):
        super().__init__(parallel=False,
                         index=index,
                         target=self.learn_manifold,
                         kwargs=kwargs)

    def learn_manifold(self, normalized_data):
        """
        :param normalized_data: array-like object that has hopefully been normalized
        """
        tsne = moboTSNE()
        tsne_cols = tsne.fit_transform(normalized_data)
        self.set_persistent(key='tsne_columns',
                            value=tsne_cols)
