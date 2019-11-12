

class Parameter(object):
    """A continuous variable with optional constraints.
    
    Args:
        name: Name of the parameter.
        lower_bound: Minimum value of the parameter.
        upper_bound: Maximum value of the parameter.
    """
    def __init__(self, name: str, lower_bound: float, 
                 upper_bound: float) -> None:
        self.name = name
        if lower_bound >= upper_bound:
            err = "`lower_bound` must be strictly less than `upper_bound`."
            raise ValueError(err)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
