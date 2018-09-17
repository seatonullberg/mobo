from mobo.engines import Task
from mobo.manifold_learning import moboTSNE


class ManifoldTaskTSNE(Task):

    def __init__(self, index):
        super().__init__(parallel=False,
                         index=index,
                         target=self.learn_manifold)

    def learn_manifold(self, data=None, data_key='normalized_data'):
        '''
        :param data: array-like object
        :param data_key: str used to retrieve data from persistent database dict
        '''
        if data is None:
            data = self.get_persistent(data_key)

        tsne = moboTSNE(config=self.configuration)
        tsne_cols = tsne.fit_transform(data)

        self.set_persistent(key='tsne_columns',
                            value=tsne_cols)
