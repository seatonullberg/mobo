from mobo.engines import Task

from sklearn.metrics import mean_squared_error
import numpy as np


class RootMeanSquaredErrorTask(Task):

    def __init__(self, index, kwargs, parallel=False, target=None):
        if target is None:
            target = self.rms_error
        super().__init__(parallel=parallel,
                         index=index,
                         target=target,
                         kwargs=kwargs)

    def rms_error(self, actual, experimental):
        """
        :param actual: list of array-like objects containing the actual values
        :param experimental: list of array-like objects containing the predicted values
        """
        assert len(actual) == len(experimental)
        errors = []
        for a, e in zip(actual, experimental):
            print(a.shape)
            print(e.shape)
            assert a.shape == e.shape
            rmse = np.sqrt(mean_squared_error(y_true=a, y_pred=e))
            errors.append(rmse)
        self.set_persistent(key='rms_errors',
                            value=errors)
