from mobo.engines import Task
from sklearn.preprocessing import StandardScaler


class StandardNormalizationTask(Task):

    def __init__(self, index):
        super().__init__(parallel=False,
                         index=index,
                         target=self.normalize)

    def normalize(self, data=None, data_key='data'):
        '''
        :param data: an array like object
        :param data_key: str used to retrieve data by a keyword
                         from the persistent database in TaskEngine
        '''
        scaler = StandardScaler()
        if data is None:
            data = self.get_persistent(data_key)
        norm_data = scaler.fit_transform(data)
        # store an array of normalized data
        self.set_persistent(key='normalized_data',
                            value=norm_data)
