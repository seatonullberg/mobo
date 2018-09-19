from mobo.engines import Task
from sklearn.preprocessing import StandardScaler


class StandardNormalizationTask(Task):

    def __init__(self, index, kwargs, parallel=False, target=None):
        if target is None:
            target = self.normalize
        super().__init__(parallel=parallel,
                         index=index,
                         target=target,
                         kwargs=kwargs)

    def normalize(self, data):
        '''
        :param data: an array like object
        '''
        scaler = StandardScaler()
        norm_data = scaler.fit_transform(data)
        # store an array of normalized data
        self.set_persistent(key='normalized_data',
                            value=norm_data)
