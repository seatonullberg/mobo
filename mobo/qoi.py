from typing import Callable


class QoI(object):
    """A quantity of interest to be evaluated during optimization.
    
    Notes:
        - `evaluator` should expect a dict of parameter names, values.

    Args:
        evaluator: Evaluation function. 
        name: Name of the qoi.
        target: Target value of the qoi.
    """
    def __init__(self, evaluator: Callable, name: str, target: float) -> None:
        self.evaluator = evaluator
        self.name = name
        self.target = target
