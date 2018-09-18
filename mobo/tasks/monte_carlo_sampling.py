from mobo.engines import Task

from scipy.stats import gaussian_kde


class KDEMonteCarloTask(Task):

    def __init__(self, index, kwargs):
        super().__init__(parallel=False,
                         index=index,
                         target=self.sample,
                         kwargs=kwargs)

    def sample(self, data, bandwidth, n_samples):
        """
        :param data: an array-like object or list of array-like objects
        :param bandwidth: list of ints or int to estimate KDE bandwidth
        :param n_samples: int to indicate how many samples to pull from data
        """
        if type(data) == list and type(bandwidth) == list:
            samples = []
            for b, d in zip(bandwidth, data):
                s = self._sample(d, n_samples, b)
                samples.append(s)
        elif type(data) != list and type(bandwidth) == list:
            raise TypeError("incompatible types for data and bandwidth")
        elif type(data) == list and type(bandwidth) != list:
            raise TypeError("incompatible types for data and bandwidth")
        else:
            # both data and bandwidth are individuals
            samples = self._sample(data, n_samples, bandwidth)

        self.set_persistent(key='kde_samples',
                            value=samples)

    def _sample(self, data, n_samples, bandwidth):
        kde = gaussian_kde(data, bandwidth)
        samples = kde.resample(n_samples)
        return samples
