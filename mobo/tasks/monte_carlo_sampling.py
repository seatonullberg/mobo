from mobo.engines import Task


class MonteCarloTask(Task):

    def __init__(self, index):
        super().__init__(parallel=True,
                         index=index,
                         target=self.sample)

    # TODO
    def sample(self, df):
        pass