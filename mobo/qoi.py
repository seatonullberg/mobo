

class QoI(object):
    """Implementation of a quantity of interest to evaluate.

    Args:
        name (str): Name of the qoi.
        target (float): Target value to optimize towards.
        evaluator (function): Evaluation function.
    """

    def __init__(self, name, target, evaluator):
        assert type(name) is str
        assert type(target) is float
        assert callable(evaluator)
        self._name = name
        self._target = target
        self._evaluator = evaluator

    @property
    def name(self):
        return self._name

    @property
    def target(self):
        return self._target

    @property
    def evaluator(self):
        return self._evaluator

