from mobo.engines import Task

from sklearn.metrics import mean_squared_error
import numpy as np


class RootMeanSquaredErrorTask(Task):

    def __init__(self, index, kwargs):
        super().__init__(parallel=False,
                         index=index,
                         target=self.rms_error,
                         kwargs=kwargs)

    def rms_error(self, actual, experimental):
        """
        :param actual: array-like object containing the actual values
        :param experimental: array-like object containing the predicted values
        """
        rmse = np.sqrt(mean_squared_error(y_true=actual, y_pred=experimental))
        self.set_persistent(key='rms_error',
                            value=rmse)
