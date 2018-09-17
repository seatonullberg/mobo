from mobo.engines import Task
from mobo.pareto import calculate_pareto


class FilterParetoTask(Task):

    def __init__(self, index):
        super().__init__(parallel=False,
                         index=index,
                         target=self.filter)

    def filter(self, data=None, data_key='sample_data'):
        '''
        :param data: array-like object
        :param data_key: str used to retrieve data from database
        '''
        if data is None:
            data = self.get_persistent(data_key)

        mask = calculate_pareto(data)
        data = data[mask]

        self.set_persistent(key='pareto_data',
                            value=data)
