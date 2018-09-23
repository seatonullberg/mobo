from mobo.engines import Task

import numpy as np


class RootMeanSquaredErrorTask(Task):

    def __init__(self, kwargs, target=None):
        if target is None:
            target = self.rms_error
        super().__init__(target=target,
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
                err_row = []
                for x in range(a.shape[1]):
                    err = (a[i][x] - e[i][x])**2
                    err = np.sqrt(err)
                    err_row.append(err)
                errors.append(np.hstack(err_row))
        errors = np.vstack(errors)
        self.set_persistent(key='rms_errors',
                            value=errors)
