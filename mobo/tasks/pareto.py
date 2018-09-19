from mobo.engines import Task
from mobo.pareto import calculate_pareto


class FilterParetoTask(Task):

    def __init__(self, index, kwargs, parallel=False, target=None):
        if target is None:
            target = self.filter
        super().__init__(parallel=parallel,
                         index=index,
                         target=target,
                         kwargs=kwargs)

    def filter(self, data):
        """
        :param data: array-like object
        """
        mask = calculate_pareto(data)
        data = data[mask]
        self.set_persistent(key='pareto_data',
                            value=data)
