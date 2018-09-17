from mobo.engines import Task
from mobo.kde import silverman_h, chiu_h


class KDETask(Task):

    def __init__(self, index, data_key=None, args=None):
        super().__init__(parallel=False,
                         index=index,
                         target=self.calculate_kde,
                         data_key=data_key,
                         args=args)

    def calculate_kde(self, data):
        """
        :param data: an array-like object
        """
        pass
