from mobo.engines import Task
from mobo.pareto import calculate_pareto


class FilterParetoTask(Task):

    def __init__(self, index, data_key=None, args=None):
        super().__init__(parallel=False,
                         index=index,
                         target=self.filter,
                         data_key=data_key,
                         args=args)

    def filter(self, data):
        """
        :param data: array-like object
        """
        mask = calculate_pareto(data)
        data = data[mask]
        self.set_persistent(key='pareto_data',
                            value=data)
