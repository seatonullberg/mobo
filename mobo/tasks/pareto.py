from mobo.engines import Task
from mobo.pareto import calculate_pareto


class FilterParetoTask(Task):

    def __init__(self, kwargs, target=None):
        if target is None:
            target = self.filter
        super().__init__(target=target,
                         kwargs=kwargs)

    def filter(self, data_to_filter, data_to_apply):
        """
        :param data_to_filter: array-like object that will be used to produce pareto mask
        :param data_to_apply: array-like object that will have the mask applied to it
        """
        print("Pareto: {x} {y}".format(x=data_to_filter.shape, y=data_to_apply.shape))
        assert data_to_filter.shape == data_to_apply.shape
        mask = calculate_pareto(data_to_filter)
        data = data_to_apply[mask]
        self.set_persistent(key='pareto_data',
                            value=data)
