from mobo.engines import Task


class MonteCarloTask(Task):

    def __init__(self, index, data_key=None, args=None):
        super().__init__(parallel=True,
                         index=index,
                         target=self.sample,
                         data_key=data_key,
                         args=args)

    def sample(self, data):
        """
        :param data: an array-like object
        """
        pass
