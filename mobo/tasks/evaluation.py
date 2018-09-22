from mobo.engines import Task

from sklearn.metrics import mean_squared_error
import numpy as np


class RootMeanSquaredErrorTask(Task):

    def __init__(self, kwargs, parallel=False, target=None):
        if target is None:
            target = self.rms_error
        super().__init__(parallel=parallel,
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
            print("Evaluation: {x} {y}".format(x=a.shape, y=e.shape))
            assert a.shape == e.shape
            for i in range(a.shape[0]):
                rmse = np.sqrt(mean_squared_error(y_true=a[i], y_pred=e[i]))
                errors.append(rmse)
        errors = np.vstack(errors)
        self.set_persistent(key='rms_errors',
                            value=errors)
