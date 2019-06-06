from mobo.parameter import Parameter
from mobo.qoi import QoI
import pandas as pd


class OptimizationData(object):
    """Underlying data collected throughout the optimization process.
    
    Args:
        parameters (iterable of Parameter): The parameters to optimize.
        qois (iterable of QoI): The quantities of interest to evaluate.
    """

    def __init__(self, parameters, qois):
        for p in parameters:
            assert type(p) is Parameter
        for q in qois:
            assert type(q) is QoI
        p_names = ["p."+p.name for p in parameters]  # parameter names
        q_names = ["q."+q.name for q in qois]  # qoi names
        e_names = ["e."+q.name for q in qois]  # error names
        self._df = pd.DataFrame(columns=p_names+q_names+e_names)
        self._parameters = parameters
        self._qois = qois

    @property
    def df(self):
        return self._df

    @property
    def parameters(self):
        return self._parameters

    @property
    def qois(self):
        return self._qois

