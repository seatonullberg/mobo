from mobo.engines import Task

import numpy as np
from scipy.stats import gaussian_kde


class KDEMonteCarloTask(Task):

    def __init__(self, kwargs, parallel=False, target=None):
        if target is None:
            target = self.sample
        super().__init__(parallel=parallel,
                         target=target,
                         kwargs=kwargs)

    def sample(self, data, bandwidth, n_samples):
        """
        :param data: an array-like object
        :param bandwidth: int used to calculate KDE
        :param n_samples: int to indicate how many samples to pull from data
        """
        print("MonteCarlo: {}".format(data.shape))
        # both data and bandwidth are individuals
        samples = []
        for n in range(n_samples):
            s = self._sample(data, bandwidth)
            samples.append(s)
        samples = np.vstack(samples)
        self.set_persistent(key='kde_samples',
                            value=samples)

    def _sample(self, data, bandwidth):
        kde = gaussian_kde(data.T, bandwidth)
        sample = kde.resample(1)
        return sample.T
