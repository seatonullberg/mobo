from typing import Optional


class Parameter(object):
    """A continuous variable with optional constraints.
    
    Args:
        name: Name of the parameter.
        fixed_bound: Static value to assign the parameter.
        lower_bound: Minimum value of the parameter.
        upper_bound: Maximum value of the parameter.
    """
    def __init__(self, name: str,
                 fixed_bound: Optional[float] = None,
                 lower_bound: Optional[float] = None,
                 upper_bound: Optional[float] = None) -> None:
        self.name = name
        self.fixed_bound = fixed_bound
        if lower_bound is not None and upper_bound is not None:
            if lower_bound >= upper_bound:
                err = (
                    "`lower_bound` must be strictly less than"
                    " `upper_bound`."
                )
                raise ValueError(err)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
