from mobo.engines import Task
from mobo.kde import silverman_h, chiu_h


class KDETask(Task):

    def __init__(self, index):
        super().__init__(parallel=False,
                         index=index,
                         target=self.calculate_kde)

    def calculate_kde(self, data=None, data_key='sample_data'):
        '''
        :param data:
        :param data_key:
        :return:
        '''
        pass
