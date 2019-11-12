from typing import Callable


class QoI(object):
    """A quantity of interest to be evaluated during optimization.
    
    Notes:
        - `evaluator` should expect a dict mapping parameter names to values.

    Args:
        name: Name of the qoi.
        evaluator: Evaluation function. 
        target: Target value of the qoi.
    """
    def __init__(self, 
                 name: str,
                 evaluator: Callable, 
                 target: float) -> None:
        self.name = name
        self.evaluator = evaluator
        self.target = target