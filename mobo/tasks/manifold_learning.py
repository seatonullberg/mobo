from mobo.engines import Task
from mobo.manifold_learning import moboTSNE


class ManifoldTaskTSNE(Task):

    def __init__(self, index):
        super().__init__(parallel=False,
                         index=index,
                         target=self.learn_manifold)

    def learn_manifold(self, df):
        '''
        :param df: pandas.DataFrame object
        :return: tSNE dimensions of the df
        '''
        tsne = moboTSNE(config=self.configuration)
        tsne_dims = tsne.fit_transform(df)
        return tsne_dims
